
import torch
import numpy as np
import os
from legged_gym.perception.surface_geometry import Cylinder, Cuboid, Sphere, Box, Ellipsoid, CylinderWell
from legged_gym.perception.ycb_geometry import YCBManager, YCBGeometry
from isaacgym.torch_utils import *
from legged_gym.perception.tracker import PCATargetTracker
from legged_gym import LEGGED_GYM_ROOT_DIR

class ObjectManager:
    """
    Manages object properties, generation, and state updates for the LeggedRobotNav environment.
    This class handles the initialization of geometric shapes (primitives and YCB),
    and the logic for resetting object properties (shape type, dimensions, task type) per episode.
    """
    def __init__(self, cfg, device, num_envs, gripper_width, camera_params_dict=None, ycb_root=None):
        self.device = device
        self.cfg = cfg
        self.target_cfg = cfg.target
        self.num_envs = num_envs
        self.gripper_width = gripper_width
        self.camera_params_dict = camera_params_dict
        self.ycb_root = ycb_root if ycb_root else os.path.join(LEGGED_GYM_ROOT_DIR, "obj_set")
        
        self.shape_types_list = self.target_cfg.shape.types
        
        self._init_buffers()
        self._init_geometry_managers()

    def _init_buffers(self):
        """ Initialize buffers for object properties """
        # Dimensions and Points
        self.object_dims = torch.zeros(self.num_envs, 3, device=self.device)
        self.ycb_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.local_points = torch.zeros(self.num_envs, self.target_cfg.perception.num_sample_points, 3, device=self.device)
        self.local_normals = torch.zeros(self.num_envs, self.target_cfg.perception.num_sample_points, 3, device=self.device)
        
        # Object Pose
        self.object_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.object_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.object_quat[:, 3] = 1.0 # Identity

        # Local Axes
        self.x_axis_local = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self.y_axis_local = torch.tensor([0.0, 1.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self.z_axis_local = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)

        # Shape Types
        self.env_shape_type_indices = torch.randint(0, len(self.shape_types_list), (self.num_envs,), device=self.device)

        # Object State Flags
        self.obj_is_vertical = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.obj_is_too_long = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.object_z_offset = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.max_dim_vals = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.max_dim_inds = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # Task Flags
        self.task_flags = torch.zeros(self.num_envs, self.cfg.env.num_task_flags, dtype=torch.float, device=self.device)
        self.is_place = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_pick = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Resampling Buffers
        if hasattr(self.cfg.commands, 'resample'):
             ranges = self.cfg.commands.resample.ranges
             self.t_min_default = torch.ones((self.num_envs, 1), device=self.device) * ranges.min_dist
             self.t_max_default = torch.ones((self.num_envs, 1), device=self.device) * ranges.max_dist
             self.dir_cam_z = torch.ones((self.num_envs, 1), device=self.device)

    def _init_geometry_managers(self):
        """ Initialize Geometry Providers (Shapes and YCB) """
        # Initialize YCB Manager
        self.ycb_manager = YCBManager(
            obj_root=self.ycb_root, 
            device=self.device,
            max_cache_points=self.target_cfg.perception.num_sample_points
        )

        # Instantiate Shape Generators
        self.shapes = {
            "cylinder": Cylinder(self.device),
            "cuboid": Cuboid(self.device),
            "sphere": Sphere(self.device),
            "box": Box(self.device),
            "ellipsoid": Ellipsoid(self.device),
            "ycb": YCBGeometry(self.ycb_manager),
            "cylinder_well": CylinderWell(self.device)
        }

    def reset_object_shapes(self, env_ids):
        """
        Main entry point for resetting/generating objects for specific environments.
        pipeline:
        1. Task Sampling (Pick vs Place)
        2. Shape Type Selection
        3. Parameter Generation & Point Sampling
        4. State Flag Updates (Isotropic, Vertical, TooLong, etc.)
        """
        if len(env_ids) == 0:
            return

        # 1. Task Sampling
        self._sample_task_types(env_ids)

        # 2. Shape Type Selection
        self._select_shape_types(env_ids)
        
        # 3. Parameter Generation
        self._sample_shape_parameters(env_ids)
            
        # 4. Update Object State Flags
        self._update_object_state_flags(env_ids)
        
        # 5. Compute Axis End Points
        self.compute_object_axis_end_points_in_world()

    def _sample_task_types(self, env_ids):
        """ Determine if the task is Pick or Place for each environment """
        target_cfg = self.target_cfg
        probs = torch.rand(len(env_ids), device=self.device)
        is_place_local = probs < target_cfg.init.place_prob # shape: [len(env_ids)]
        
        self.is_place[env_ids] = is_place_local
        self.is_pick[env_ids] = ~is_place_local
        self.task_flags[env_ids] = is_place_local.float().unsqueeze(-1)

    def _select_shape_types(self, env_ids):
        """ Select shape types for each environment, overriding for Place tasks """
        # 1. Random Selection (Default: Pick friendly)
        # Avoid "box" for base pick tasks if possible to ensure variety, or keep it random.
        # Original logic: "Select non-box types for all environments initially to ensure Pick tasks don't get Box"
        other_indices = [i for i, name in enumerate(self.shape_types_list) if name != "box"]
        if other_indices:
            other_tensor = torch.tensor(other_indices, device=self.device)
            # Randomly pick from non-box types
            type_indices = other_tensor[torch.randint(0, len(other_indices), (len(env_ids),), device=self.device)]
        else:
            type_indices = torch.randint(0, len(self.shape_types_list), (len(env_ids),), device=self.device)
        
        # 2. Override for Place Task
        is_place_local = self.is_place[env_ids]
        if is_place_local.any():
            place_type_indices = []
            if "box" in self.shape_types_list:
                place_type_indices.append(self.shape_types_list.index("box"))
            if "cylinder_well" in self.shape_types_list:
                place_type_indices.append(self.shape_types_list.index("cylinder_well"))
                
            if len(place_type_indices) > 0:
                idx_selector = torch.randint(0, len(place_type_indices), (len(env_ids),), device=self.device)
                chosen_place_indices = torch.tensor(place_type_indices, device=self.device)[idx_selector]
                
                # Apply only to place tasks
                type_indices[is_place_local] = chosen_place_indices[is_place_local]
            else:
                # Fallback: No placeable shapes available -> Convert to Pick
                self.is_place[env_ids] = False
                self.is_pick[env_ids] = True
                self.task_flags[env_ids] = 0.
            
        self.env_shape_type_indices[env_ids] = type_indices

    def _sample_shape_parameters(self, env_ids):
        """ Sample shape dimensions and surface points """
        target_cfg = self.target_cfg
        is_place_local = self.is_place[env_ids]

        # Iterate over types
        for i, type_name in enumerate(self.shape_types_list):
            subset_indices = self.env_shape_type_indices[env_ids]
            mask = (subset_indices == i)
            
            if not mask.any():
                continue
            
            current_ids = env_ids[mask]
            # subset of is_place_local
            current_is_place = is_place_local[mask] 
            count = len(current_ids)
            
            # Generate Raw Params (radius, height, or dims)
            params = self._generate_shape_params(type_name, count, current_ids, current_is_place)
            
            if type_name == "ycb":
                # YCB params logic is handled inside _generate_shape_params (sampling IDS, getting info)
                # But we need to update buffers if not done there. 
                # Actually _generate_shape_params for YCB returns None and does side effects in current logic
                # Let's verify _generate_shape_params implementation below.
                continue 

            # Compute Object Dimensions (Bounding Box)
            dims = PCATargetTracker.compute_object_dims(type_name, params, self.device)
            
            # Sample Surface Points for Point Cloud
            shape = self.shapes[type_name]
            pts, nrms = shape.sample_surface(target_cfg.perception.num_sample_points, count, params)
            
            # Update Environment Buffers
            self.object_dims[current_ids] = dims
            self.local_points[current_ids] = pts
            self.local_normals[current_ids] = nrms
            
        # Update tracker logic moved to caller (LeggedRobotNav)

            
    def _generate_shape_params(self, type_name, count, current_ids, current_is_place):
        """ Helper to generate random parameters for a specific shape type """
        target_cfg = self.target_cfg
        
        if type_name == "cylinder":
            r = torch_rand_float(target_cfg.shape.radius_range[0], target_cfg.shape.radius_range[1], (count, 1), device=self.device)
            h = torch_rand_float(target_cfg.shape.height_range[0], target_cfg.shape.height_range[1], (count, 1), device=self.device)
            return torch.cat([r, h], dim=1)
            
        elif type_name == "cylinder_well":
            # Scale radius from box dims x
            r_box_min = target_cfg.shape.box_dims_range[0][0]
            r_box_max = target_cfg.shape.box_dims_range[0][1]
            r = torch_rand_float(r_box_min/2.0, r_box_max/2.0, (count, 1), device=self.device)
            
            h_min = target_cfg.shape.box_dims_range[2][0]
            h_max = target_cfg.shape.box_dims_range[2][1]
            h = torch_rand_float(h_min, h_max, (count, 1), device=self.device)
            return torch.cat([r, h], dim=1)
            
        elif type_name == "cuboid" or type_name == "box":
            # Sample Small (Pick)
            x_s = torch_rand_float(target_cfg.shape.dims_range[0][0], target_cfg.shape.dims_range[0][1], (count, 1), device=self.device)
            y_s = torch_rand_float(target_cfg.shape.dims_range[1][0], target_cfg.shape.dims_range[1][1], (count, 1), device=self.device)
            z_s = torch_rand_float(target_cfg.shape.dims_range[2][0], target_cfg.shape.dims_range[2][1], (count, 1), device=self.device)
            
            # Sample Large (Place)
            x_l = torch_rand_float(target_cfg.shape.box_dims_range[0][0], target_cfg.shape.box_dims_range[0][1], (count, 1), device=self.device)
            y_l = torch_rand_float(target_cfg.shape.box_dims_range[1][0], target_cfg.shape.box_dims_range[1][1], (count, 1), device=self.device)
            z_l = torch_rand_float(target_cfg.shape.box_dims_range[2][0], target_cfg.shape.box_dims_range[2][1], (count, 1), device=self.device)
            
            # Select based on task
            x = torch.where(current_is_place.unsqueeze(-1), x_l, x_s)
            y = torch.where(current_is_place.unsqueeze(-1), y_l, y_s)
            z = torch.where(current_is_place.unsqueeze(-1), z_l, z_s)
            return torch.cat([x, y, z], dim=1)
            
        elif type_name == "sphere":
            return torch_rand_float(target_cfg.shape.radius_range[0], target_cfg.shape.radius_range[1], (count, 1), device=self.device)
            
        elif type_name == "ellipsoid":
            x = torch_rand_float(target_cfg.shape.ellipsoid_dims_range[0][0], target_cfg.shape.ellipsoid_dims_range[0][1], (count, 1), device=self.device)
            y = torch_rand_float(target_cfg.shape.ellipsoid_dims_range[1][0], target_cfg.shape.ellipsoid_dims_range[1][1], (count, 1), device=self.device)
            z = torch_rand_float(target_cfg.shape.ellipsoid_dims_range[2][0], target_cfg.shape.ellipsoid_dims_range[2][1], (count, 1), device=self.device)
            return torch.cat([x, y, z], dim=1)
            
        elif type_name == "ycb":
            num_ycb = self.ycb_manager.get_num_objects()
            if num_ycb > 0:
                indices = torch.randint(0, num_ycb, (count,), device=self.device)
            else:
                indices = torch.zeros(count, dtype=torch.long, device=self.device)
            
            self.ycb_indices[current_ids] = indices
            extents, _ = self.ycb_manager.get_info(indices)
            self.object_dims[current_ids] = extents
            pts, nrms = self.shapes["ycb"].sample_surface(target_cfg.perception.num_sample_points, count, indices)
            self.local_points[current_ids] = pts
            self.local_normals[current_ids] = nrms
            return None # YCB handled directly

        return None

    def _update_object_state_flags(self, env_ids):
        """ Update object state flags like vertical/horizontal, size, long/short logic etc. """
        self.max_dim_vals[env_ids], self.max_dim_inds[env_ids] = torch.max(self.object_dims[env_ids], dim=1)
        
        # 1. Determine Vertical/Horizontal
        random_vertical = torch.rand(len(env_ids), device=self.device) < self.target_cfg.init.vertical_prob
        
        # Force Place Task to be Horizontal (Standard orientation)
        is_place_local = self.is_place[env_ids]
        random_vertical[is_place_local] = False
        
        self.obj_is_vertical[env_ids] = random_vertical
        
        # [Fix] Force YCB objects to be Vertical
        if "ycb" in self.shape_types_list:
            ycb_idx = self.shape_types_list.index("ycb")
            is_ycb = (self.env_shape_type_indices[env_ids] == ycb_idx)
            self.obj_is_vertical[env_ids][is_ycb] = True
        
        is_horizontal = ~self.obj_is_vertical[env_ids]
        is_large = self.max_dim_vals[env_ids] > self.gripper_width

        # 2. Identify "Round-ish" Objects (Isotropic)
        sorted_dims, _ = torch.sort(self.object_dims[env_ids], dim=1, descending=True)
        # If ratio of Intermediate / Longest > 0.6, considered isotropic
        is_isotropic = (sorted_dims[:, 1] / sorted_dims[:, 0]) > 0.6
        
        # 3. Determine "Too Long" (triggers Long Logic)
        # Too Long = Horizontal AND Large AND Not Isotropic
        self.obj_is_too_long[env_ids] = is_horizontal & is_large & (~is_isotropic)
        
        # 4. Compute Z Offset (Height above ground)
        z_off = torch.zeros_like(self.object_dims[env_ids, 2])
        self._compute_z_offsets(env_ids, z_off)
        self.object_z_offset[env_ids] = z_off

    def _compute_z_offsets(self, env_ids, z_off):
        """ Compute Z offsets based on type and orientation """
        # Only iterate over types present in the list
        for type_name in self.shape_types_list:
             # Safety check if shape exists
             if type_name not in self.shapes: continue
             
             type_idx = self.shape_types_list.index(type_name)
             mask_type = (self.env_shape_type_indices[env_ids] == type_idx)
             
             if not mask_type.any(): continue
             
             # Sub-mask for orientation
             mask_v = mask_type & self.obj_is_vertical[env_ids]
             mask_h = mask_type & (~self.obj_is_vertical[env_ids])
             
             dims = self.object_dims[env_ids]
             
             if type_name == "cylinder":
                 # Vertical: h/2 (dims[2]/2), Horizontal: d/2 (dims[0]/2)
                 if mask_v.any(): z_off[mask_v] = dims[mask_v, 2] / 2.0
                 if mask_h.any(): z_off[mask_h] = dims[mask_h, 0] / 2.0
                 
             elif type_name in ["cuboid", "box"]:
                 # Vertical: y/2 (dims[1]/2), Horizontal: z/2 (dims[2]/2)
                 if mask_v.any(): z_off[mask_v] = dims[mask_v, 1] / 2.0
                 if mask_h.any(): z_off[mask_h] = dims[mask_h, 2] / 2.0
                 
             elif type_name == "cylinder_well":
                  if mask_type.any(): z_off[mask_type] = dims[mask_type, 2] / 2.0
                  
             elif type_name == "sphere":
                  if mask_type.any(): z_off[mask_type] = dims[mask_type, 2] / 2.0
                  
             elif type_name == "ellipsoid":
                  if mask_type.any(): z_off[mask_type] = dims[mask_type, 2] / 2.0
                  
             elif type_name == "ycb":
                  if mask_type.any(): z_off[mask_type] = dims[mask_type, 2] / 2.0

    def resample_object_positions(self, env_ids, cam_transform, root_states):
        """ set the goal position in the visualable zone of the camera
        """
        if len(env_ids) == 0:
            return

        # 1. Sample u, v in Image Plane (Normalized [0, 1])
        u = torch.rand((len(env_ids), 1), device=self.device)
        v = torch.rand((len(env_ids), 1), device=self.device)
        
        # Unpack camera params
        cam_params = self.camera_params_dict
        fx = cam_params['fx'][env_ids]
        fy = cam_params['fy'][env_ids]
        cx = cam_params['cx'][env_ids]
        cy = cam_params['cy'][env_ids]
        img_w = cam_params['img_w'][env_ids]
        img_h = cam_params['img_h'][env_ids]
        
        # 2. Compute ray in Camera Frame (Z-forward)
        # P_cam = [x, y, z] = z * [(u - cx)/fx, (v - cy)/fy, 1]
        dir_cam_x = (u * img_w - cx) / fx
        dir_cam_y = (v * img_h - cy) / fy
        dir_cam_z = self.dir_cam_z[env_ids]
        
        dir_cam = torch.cat([dir_cam_x, dir_cam_y, dir_cam_z], dim=-1) # (N, 3)
        
        # 3. Transform ray to World Frame
        # dir_base = R^T * dir_cam
        R = cam_transform['R'][env_ids]
        dir_base = torch.bmm(R.transpose(1, 2), dir_cam.unsqueeze(-1)).squeeze(-1)
        
        base_quat = root_states[env_ids, 3:7]
        dir_world = quat_apply(base_quat, dir_base)
        
        # 4. Camera World Position
        # pos_cam_base = T
        T = cam_transform['T'][env_ids]
        base_pos = root_states[env_ids, :3]
        pos_cam_world = quat_apply(base_quat, T) + base_pos
        
        # 5. Compute valid depth range [t_min, t_max] for z_cam (which is t)
        height_min = 0.00 # 0cm above ground
        
        # Default range
        t_min = self.t_min_default[env_ids].clone()
        t_max = self.t_max_default[env_ids].clone()
        
        dz = dir_world[:, 2:3]
        pz = pos_cam_world[:, 2:3]
        
        # Case 1: Ray points down (dz < -1e-4)
        mask_down = dz < -1e-4
        t_limit_down = (height_min - pz) / dz
        t_max = torch.where(mask_down, torch.min(t_max, t_limit_down), t_max)
        
        # Case 2: Ray points up (dz > 1e-4)
        mask_up = dz > 1e-4
        t_limit_up = (height_min - pz) / dz
        t_min = torch.where(mask_up, torch.max(t_min, t_limit_up), t_min)
        
        # Ensure t_min <= t_max (prioritize height constraint t_max)
        t_min = torch.min(t_min, t_max)

        # Sample t
        rand_val = torch.rand((len(env_ids), 1), device=self.device)
        t = t_min + rand_val * (t_max - t_min)
        
        # 6. Compute P_world
        P_world = pos_cam_world + t * dir_world

        # Use cached z_offset
        P_world[:, 2] = self.object_z_offset[env_ids]
        
        # Update Internal State
        self.object_pos[env_ids] = P_world

    def resample_object_orientations(self, env_ids):
        """ set the goal orientation
        """
        if len(env_ids) == 0:
            return
        
        # 1. Compute Yaw (Random or Fixed)
        if self.target_cfg.init.randomize_orientation:
            yaw = torch.rand((len(env_ids), 1), device=self.device) * 2 * np.pi
        else:
            yaw = torch.zeros((len(env_ids), 1), device=self.device)
            
        # 2. Convert to Quaternion (Rotation around Z)
        sy = torch.sin(yaw * 0.5)
        cy = torch.cos(yaw * 0.5)
        q_yaw = torch.cat([torch.zeros_like(sy), torch.zeros_like(sy), sy, cy], dim=-1)
        
        # 3. Apply shape-specific base rotation
        q_final = q_yaw.clone()
        
        # Identify shapes
        is_cylinder = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        is_cuboid = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        
        if "cylinder" in self.shape_types_list:
            cyl_idx = self.shape_types_list.index("cylinder")
            is_cylinder = (self.env_shape_type_indices[env_ids] == cyl_idx)
            
        if "cuboid" in self.shape_types_list:
            cub_idx = self.shape_types_list.index("cuboid")
            is_cuboid = (self.env_shape_type_indices[env_ids] == cub_idx)
            
        is_vertical = self.obj_is_vertical[env_ids]
        is_horizontal = ~is_vertical
        
        # Prepare Rotations
        val = np.sin(np.pi / 4)
        # q_rot_y_90: Rotate 90 deg around Y (Z -> X)
        q_rot_y_90 = torch.tensor([0.0, val, 0.0, val], device=self.device, dtype=torch.float).view(1, 4).repeat(len(env_ids), 1)
        
        # q_rot_x_90: Rotate 90 deg around X (Y -> Z)
        q_rot_x_90 = torch.tensor([val, 0.0, 0.0, val], device=self.device, dtype=torch.float).view(1, 4).repeat(len(env_ids), 1)
        
        # Apply Cylinder Logic
        # Horizontal Cylinder -> Rotate Y 90 (Lay down)
        mask_cyl_horz = is_cylinder & is_horizontal
        if mask_cyl_horz.any():
            q_final[mask_cyl_horz] = quat_mul(q_yaw[mask_cyl_horz], q_rot_y_90[mask_cyl_horz])
            
        # Apply Cuboid Logic
        # Vertical Cuboid -> Rotate X 90 (Stand up, assuming Y is long)
        mask_cub_vert = is_cuboid & is_vertical
        if mask_cub_vert.any():
            q_final[mask_cub_vert] = quat_mul(q_yaw[mask_cub_vert], q_rot_x_90[mask_cub_vert])

        self.object_quat[env_ids] = q_final

    def compute_object_axis_end_points_in_world(self):
        """ Compute object axes and head/tail points in world frame.
            Derived from current object_pos and object_quat.
            Stores results in internal buffers.
        """
        # Object axes in world frame
        x_axis_world = quat_apply(self.object_quat, self.x_axis_local)
        y_axis_world = quat_apply(self.object_quat, self.y_axis_local)
        z_axis_world = quat_apply(self.object_quat, self.z_axis_local)
        
        # Store if needed, or return?
        # The prompt says ObjectManager should manage its own state.
        # But LeggedRobotNav uses these.
        # I'll store them in ObjectManager.
        self.x_axis_world = x_axis_world
        self.y_axis_world = y_axis_world
        self.z_axis_world = z_axis_world
        self.all_axes = torch.stack([x_axis_world, y_axis_world, z_axis_world], dim=1)

        batch_indices = torch.arange(self.num_envs, device=self.device)
        self.axis_long_world = self.all_axes[batch_indices, self.max_dim_inds] # [N, 3]
        
        half_length = (self.max_dim_vals / 2.0).unsqueeze(-1)
        self.p_head_world = self.object_pos + self.axis_long_world * half_length
        self.p_tail_world = self.object_pos - self.axis_long_world * half_length

    def resample_object_pose_near_success(self, env_ids, robot_pos, robot_quat, dist_lat_range):
        """
        Resample object pose such that the robot is on the path between Hint and Optimal.
        This "feeds" the robot a near-success state.
        """
        if len(env_ids) == 0:
            return

        # 1. Get Robot Position (Gripper)
        # robot_pos passed from args
        
        # 2. Determine Direction
        # Use robot's current heading (Yaw) to define the path direction
        q_robot = robot_quat
        _, _, yaw = get_euler_xyz(q_robot)
        
        # Direction vector (XY plane)
        dir_x = torch.cos(yaw)
        dir_y = torch.sin(yaw)
        dir_z = torch.zeros_like(dir_x)
        path_dir = torch.stack([dir_x, dir_y, dir_z], dim=-1) # [N, 3]
        
        # 3. Handle Place Tasks
        is_place_local = self.is_place[env_ids]
        if is_place_local.any():
            self._resample_near_success_place_object(env_ids, is_place_local, robot_pos, path_dir, yaw, dist_lat_range)
            
        # 4. Handle Pick Tasks
        is_pick_local = self.is_pick[env_ids]
        if is_pick_local.any():
            # Split into Long and Short
            long_mask = self.obj_is_too_long[env_ids] & is_pick_local
            short_mask = (~self.obj_is_too_long[env_ids]) & is_pick_local
            
            if long_mask.any():
                self._resample_near_success_long_object(env_ids, long_mask, robot_pos, path_dir, yaw, dist_lat_range)
            
            if short_mask.any():
                self._resample_near_success_short_object(env_ids, short_mask, robot_pos, path_dir, yaw, dist_lat_range)

        # Update derived quantities
        self.compute_object_axis_end_points_in_world()


    def _resample_near_success_place_object(self, env_ids, mask, robot_pos, path_dir, yaw, dist_lat_range):
        """ Helper for place object resampling (Large Box) """
        ids_place = env_ids[mask]
        n_place = len(ids_place)
        
        # Radius estimate (max dim / 2)
        radius = torch.max(self.object_dims[ids_place], dim=1)[0].unsqueeze(-1) / 2.0
        
        dist_fwd = torch_rand_float(-0.1, 0.4, (n_place, 1), device=self.device)
        dist_fwd = dist_fwd + radius
        dist_lat = torch_rand_float(dist_lat_range[0], dist_lat_range[1], (n_place, 1), device=self.device)
            
        dir_place = path_dir[mask]
        
        # Compute Lateral Direction: (-sin, cos, 0)
        dir_lat = torch.stack([-dir_place[:, 1], dir_place[:, 0], torch.zeros_like(dir_place[:, 0])], dim=-1)
        
        robot_pos_place = robot_pos[mask]
        
        # New Object Pos
        obj_pos_xy = robot_pos_place[:, :2] + \
                     dir_place[:, :2] * dist_fwd + \
                     dir_lat[:, :2] * dist_lat
                     
        self.object_pos[ids_place, 0] = obj_pos_xy[:, 0]
        self.object_pos[ids_place, 1] = obj_pos_xy[:, 1]
        
        # self.env.env_start_pos updated by Caller
        
    def _resample_near_success_long_object(self, env_ids, mask, robot_pos, path_dir, yaw, dist_lat_range):
        """ Helper for long object resampling """
        ids_long = env_ids[mask]
        n_long = len(ids_long)
        
        # Randomize forward distance
        dist_fwd = torch_rand_float(0.1, 0.5, (n_long, 1), device=self.device)
        # Randomize lateral offset
        dist_lat = torch_rand_float(dist_lat_range[0], dist_lat_range[1], (n_long, 1), device=self.device)
        
        # Object Length
        L = self.max_dim_vals[ids_long].unsqueeze(-1)
        
        # Center Position: Center = Robot + Dir_Fwd * (L/2 + dist_fwd) + Dir_Lat * dist_lat
        dir_long = path_dir[mask] # [N, 3] (cos, sin, 0)
        
        # Compute Lateral Direction: (-sin, cos, 0)
        dir_lat = torch.stack([-dir_long[:, 1], dir_long[:, 0], torch.zeros_like(dir_long[:, 0])], dim=-1)
        
        robot_pos_long = robot_pos[mask]
        
        center_xy = robot_pos_long[:, :2] +                     dir_long[:, :2] * (L / 2.0 + dist_fwd) +                     dir_lat[:, :2] * dist_lat
        
        self.object_pos[ids_long, 0] = center_xy[:, 0]
        self.object_pos[ids_long, 1] = center_xy[:, 1]
        
        # Orientation: Align Long Axis with Path Dir
        # 1. Base rotation: Align Global X with Path Dir
        # Add random yaw perturbation
        yaw_noise = torch_rand_float(-20.0, 20.0, (n_long, 1), device=self.device).squeeze(-1)
        yaw_long = yaw[mask] + yaw_noise
        sy = torch.sin(yaw_long * 0.5)
        cy = torch.cos(yaw_long * 0.5)
        q_yaw = torch.stack([torch.zeros_like(sy), torch.zeros_like(sy), sy, cy], dim=-1)
        
        # 2. Correction rotation: Align Object Long Axis with Global X
        # Find which axis is the long axis
        long_axis_idx = self.max_dim_inds[ids_long] # [N]
        
        # Prepare correction quaternions
        # Case 0: X is long (Identity) -> [0, 0, 0, 1]
        q_corr_0 = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device, dtype=torch.float).repeat(n_long, 1)
        
        # Case 1: Y is long (Rotate -90 deg around Z) -> [0, 0, -0.707, 0.707]
        val_1 = np.sin(-np.pi / 4)
        c_val_1 = np.cos(-np.pi / 4)
        q_corr_1 = torch.tensor([0.0, 0.0, val_1, c_val_1], device=self.device, dtype=torch.float).repeat(n_long, 1)
        
        # Case 2: Z is long (Rotate 90 deg around Y) -> [0, 0.707, 0, 0.707]
        val_2 = np.sin(np.pi / 4)
        c_val_2 = np.cos(np.pi / 4)
        q_corr_2 = torch.tensor([0.0, val_2, 0.0, c_val_2], device=self.device, dtype=torch.float).repeat(n_long, 1)
        
        # Select based on index
        q_corr = torch.where(
            (long_axis_idx == 1).unsqueeze(-1),
            q_corr_1,
            torch.where(
                (long_axis_idx == 2).unsqueeze(-1),
                q_corr_2,
                q_corr_0
            )
        )
        
        # Combine: q_final = q_yaw * q_corr
        self.object_quat[ids_long] = quat_mul(q_yaw, q_corr)

    def _resample_near_success_short_object(self, env_ids, mask, robot_pos, path_dir, yaw, dist_lat_range):
        """ Helper for short object resampling """
        ids_short = env_ids[mask]
        n_short = len(ids_short)
        
        # Sample dist_fwd (distance to goal)
        dist_fwd = torch_rand_float(0.1, 0.4, (n_short, 1), device=self.device)
        # Sample dist_lat (lateral offset)
        dist_lat = torch_rand_float(dist_lat_range[0], dist_lat_range[1], (n_short, 1), device=self.device)
            
        dir_short = path_dir[mask]
        
        # Compute Lateral Direction: (-sin, cos, 0)
        dir_lat = torch.stack([-dir_short[:, 1], dir_short[:, 0], torch.zeros_like(dir_short[:, 0])], dim=-1)
        
        robot_pos_short = robot_pos[mask]
        
        # New Object Pos
        obj_pos_xy = robot_pos_short[:, :2] + \
                     dir_short[:, :2] * dist_fwd + \
                     dir_lat[:, :2] * dist_lat
                     
        self.object_pos[ids_short, 0] = obj_pos_xy[:, 0]
        self.object_pos[ids_short, 1] = obj_pos_xy[:, 1]
        
        # self.env.env_start_pos updated by Caller
        
        # Reset Orientation: Align Global X with Path Dir (Upright)
        yaw_short = yaw[mask]
        sy = torch.sin(yaw_short * 0.5)
        cy = torch.cos(yaw_short * 0.5)
        q_yaw = torch.stack([torch.zeros_like(sy), torch.zeros_like(sy), sy, cy], dim=-1)
        
        q_final = q_yaw.clone()
        
        # Identify Shapes
        is_cylinder = torch.zeros(n_short, dtype=torch.bool, device=self.device)
        is_cuboid = torch.zeros(n_short, dtype=torch.bool, device=self.device)
        
        if "cylinder" in self.shape_types_list:
            cyl_idx = self.shape_types_list.index("cylinder")
            is_cylinder = (self.env_shape_type_indices[ids_short] == cyl_idx)
            
        if "cuboid" in self.shape_types_list:
            cub_idx = self.shape_types_list.index("cuboid")
            is_cuboid = (self.env_shape_type_indices[ids_short] == cub_idx)
            
        is_vertical = self.obj_is_vertical[ids_short]
        is_horizontal = ~is_vertical
        
        # Prepare Rotations
        val = np.sin(np.pi / 4)
        q_rot_y_90 = torch.tensor([0.0, val, 0.0, val], device=self.device, dtype=torch.float).view(1, 4).repeat(n_short, 1)
        q_rot_x_90 = torch.tensor([val, 0.0, 0.0, val], device=self.device, dtype=torch.float).view(1, 4).repeat(n_short, 1)
        
        # Apply Cylinder Logic
        mask_cyl_horz = is_cylinder & is_horizontal
        if mask_cyl_horz.any():
            q_final[mask_cyl_horz] = quat_mul(q_yaw[mask_cyl_horz], q_rot_y_90[mask_cyl_horz])
            
        # Apply Cuboid Logic
        mask_cub_vert = is_cuboid & is_vertical
        if mask_cub_vert.any():
            q_final[mask_cub_vert] = quat_mul(q_yaw[mask_cub_vert], q_rot_x_90[mask_cub_vert])

        self.object_quat[ids_short] = q_final
