
import torch

class GraspPoseOptimizer:
    """
    Computes optimal grasp poses and hint poses based on object shape and task type.
    Decouples grasp logic from the main environment class.
    """
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        
        # Cache config values
        self.grasp_offset_long = cfg.env.grasp_offset_long
        self.hint_dist_long = cfg.env.hint_dist_long
        self.grasp_offset_short = cfg.env.grasp_offset_short
        self.hint_dist_short = cfg.env.hint_dist_short
        self.hint_dist_place = cfg.env.hint_dist_place
        self.place_pitch_target = cfg.commands.place_pitch_target
        self.place_clearance = cfg.target.shape.place_clearance
        # self.grasp_offset_place = cfg.env.grasp_offset_place # If it exists

    def compute_optimal_hint_pose(self, 
                                   robot_base_pos,
                                   gripper_pos,
                                   object_pos,
                                   object_dims,
                                   p_head_world,
                                   p_tail_world,
                                   axis_long_world,
                                   env_start_pos,
                                   obj_is_too_long,
                                   is_place,
                                   is_pick):
        """
        Main entry point to compute optimal grasp and hint poses.
        
        Args:
            robot_base_pos (Tensor): [N, 3] Robot base position
            gripper_pos (Tensor): [N, 3] Gripper position (world)
            object_pos (Tensor): [N, 3] Object center position (world)
            object_dims (Tensor): [N, 3] Object dimensions
            p_head_world (Tensor): [N, 3] Object head point (world)
            p_tail_world (Tensor): [N, 3] Object tail point (world)
            axis_long_world (Tensor): [N, 3] Object long axis vector (world)
            env_start_pos (Tensor): [N, 3] Episode start position (for hint reference)
            obj_is_too_long (Tensor): [N] Boolean mask for long objects
            is_place (Tensor): [N] Boolean mask for place task
            is_pick (Tensor): [N] Boolean mask for pick task (usually ~is_place)
            
        Returns:
            optimal_grasp_pos, optimal_grasp_quat, optimal_approach_dir, hint_pos, hint_quat
        """
        
        # 1. Compute for Long Axis Objects
        res_long = self._compute_long_object_logic(
            robot_base_pos, gripper_pos, p_head_world, p_tail_world, axis_long_world
        )
        (opt_pos_long, opt_quat_long, opt_dir_long, hint_pos_long, hint_quat_long) = res_long
        
        # 2. Compute for Short Axis Objects
        res_short = self._compute_short_object_logic(
            robot_base_pos, object_pos, object_dims, env_start_pos
        )
        (opt_pos_short, opt_quat_short, opt_dir_short, hint_pos_short, hint_quat_short) = res_short
        
        # 3. Compute for Place Task
        res_place = self._compute_place_task_logic(
            robot_base_pos, object_pos, object_dims, env_start_pos
        )
        (opt_pos_place, opt_quat_place, opt_dir_place, hint_pos_place, hint_quat_place) = res_place
        
        # 4. Selection Logic
        use_short_logic = (~obj_is_too_long).unsqueeze(-1)
        
        # Pick Task Selection (Long vs Short)
        opt_pos_pick = torch.where(use_short_logic, opt_pos_short, opt_pos_long)
        opt_quat_pick = torch.where(use_short_logic, opt_quat_short, opt_quat_long)
        opt_dir_pick = torch.where(use_short_logic, opt_dir_short, opt_dir_long)
        hint_pos_pick = torch.where(use_short_logic, hint_pos_short, hint_pos_long)
        hint_quat_pick = torch.where(use_short_logic, hint_quat_short, hint_quat_long)
        
        # Final Selection (Pick vs Place)
        is_place_expanded = is_place.unsqueeze(-1)
        
        optimal_grasp_pos = torch.where(is_place_expanded, opt_pos_place, opt_pos_pick)
        optimal_grasp_quat = torch.where(is_place_expanded, opt_quat_place, opt_quat_pick)
        optimal_approach_dir = torch.where(is_place_expanded, opt_dir_place, opt_dir_pick)
        
        hint_pos = torch.where(is_place_expanded, hint_pos_place, hint_pos_pick)
        hint_quat = torch.where(is_place_expanded, hint_quat_place, hint_quat_pick)
        
        return optimal_grasp_pos, optimal_grasp_quat, optimal_approach_dir, hint_pos, hint_quat

    def _compute_long_object_logic(self, robot_base_pos, gripper_pos, p_head_world, p_tail_world, axis_long_world):
        offset = self.grasp_offset_long
        hint_dist = self.hint_dist_long
        
        # Determine Head vs Tail
        dist_head = torch.norm(p_head_world - gripper_pos, dim=-1)
        dist_tail = torch.norm(p_tail_world - gripper_pos, dim=-1)
        head_closer = dist_head < dist_tail
        
        # Outward Axis
        outward_dir = torch.where(head_closer.unsqueeze(-1), axis_long_world, -axis_long_world)
        
        # Optimal Pose
        closest_vertex = torch.where(head_closer.unsqueeze(-1), p_head_world, p_tail_world)
        opt_pos = closest_vertex + outward_dir * offset
        opt_dir = -outward_dir
        
        # Dynamic Pitch
        vec_base_to_target = opt_pos - robot_base_pos[:, :3]
        d_xy = torch.norm(vec_base_to_target[:, :2], dim=1)
        d_z = vec_base_to_target[:, 2]
        pitch_dynamic = torch.atan2(-d_z, d_xy)
        
        # Hint Pose
        hint_pos = closest_vertex + outward_dir * hint_dist
        
        # Quaternions
        opt_quat = self._compute_quat_from_dir(opt_dir, pitch_dynamic)
        hint_quat = opt_quat.clone()
        
        return opt_pos, opt_quat, opt_dir, hint_pos, hint_quat

    def _compute_short_object_logic(self, robot_base_pos, object_pos, object_dims, env_start_pos):
        offset = self.grasp_offset_short
        hint_dist_from_start = self.hint_dist_short
        
        # Approach Axis (Start -> Object)
        vec_start_to_obj = object_pos - env_start_pos
        vec_start_to_obj[:, 2] = 0.0 # XY plane
        dist_start_to_obj = torch.norm(vec_start_to_obj, dim=-1, keepdim=True)
        axis_inward = vec_start_to_obj / (dist_start_to_obj + 1e-6)
        
        # Optimal Pose
        radius = torch.min(object_dims, dim=1)[0].unsqueeze(-1) / 2.0
        opt_pos = object_pos - axis_inward * (radius + offset)
        opt_dir = axis_inward
        
        # Dynamic Pitch
        vec_base_to_target = opt_pos - robot_base_pos[:, :3]
        d_xy = torch.norm(vec_base_to_target[:, :2], dim=1)
        d_z = vec_base_to_target[:, 2]
        pitch_dynamic = torch.atan2(-d_z, d_xy)
        
        # Hint Pose
        hint_pos = env_start_pos + axis_inward * hint_dist_from_start
        hint_pos[:, 2] = object_pos[:, 2]
        
        # Quaternions
        opt_quat = self._compute_quat_from_dir(opt_dir, pitch_dynamic)
        hint_quat = opt_quat.clone()
        
        return opt_pos, opt_quat, opt_dir, hint_pos, hint_quat

    def _compute_place_task_logic(self, robot_base_pos, object_pos, object_dims, env_start_pos):
        clearance = self.place_clearance
        hint_dist_from_start = self.hint_dist_place
        
        # Approach Axis
        vec_start_to_obj = object_pos - env_start_pos
        vec_start_to_obj[:, 2] = 0.0
        dist_start_to_obj = torch.norm(vec_start_to_obj, dim=-1, keepdim=True)
        axis_inward = vec_start_to_obj / (dist_start_to_obj + 1e-6)
        
        # Optimal Pose
        offset_sample = 0.0 # Fixed as per original code
        opt_pos = object_pos - axis_inward * offset_sample
        opt_pos[:, 2] = object_pos[:, 2] + object_dims[:, 2] / 2.0 + clearance
        opt_dir = axis_inward
        
        # Pitch
        d_xy = torch.norm((opt_pos - robot_base_pos[:, :3])[:, :2], dim=1) # approximation
        # pitch_fixed = torch.ones_like(d_xy) * self.place_pitch_target
        pitch_target = torch.full_like(d_xy, self.place_pitch_target)
        
        # Hint Pose
        hint_pos = env_start_pos + axis_inward * hint_dist_from_start
        hint_pos[:, 2] = object_pos[:, 2] + object_dims[:, 2] / 2.0 + clearance
        
        # Quaternions
        opt_quat = self._compute_quat_from_dir(opt_dir, pitch_target)
        hint_quat = opt_quat.clone()
        
        return opt_pos, opt_quat, opt_dir, hint_pos, hint_quat

    def _compute_quat_from_dir(self, direction, pitch):
        """
        Computes quaternion from direction vector and pitch angle.
        Matches logic from LeggedRobotNav.
        """
        # Yaw: atan2(dir.y, dir.x)
        yaw = torch.atan2(direction[:, 1], direction[:, 0])

        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)

        # Ensure pitch is a tensor
        if isinstance(pitch, float):
            pitch = torch.tensor(pitch, device=self.device)
        elif pitch.dim() == 0:
            pitch = pitch.unsqueeze(0).expand(yaw.shape[0])
            
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)

        # q = [-sp*sy, sp*cy, cp*sy, cp*cy] (Roll=0, Pitch=pitch, Yaw=yaw)
        # Note: This formula corresponds to q_yaw * q_pitch_around_local_y ? Let's trust the original code.
        q_x = -sp * sy
        q_y = sp * cy
        q_z = cp * sy
        q_w = cp * cy

        quat = torch.stack([q_x, q_y, q_z, q_w], dim=-1)

        # Normalize quaternion to ensure numerical stability
        quat = quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-6)

        return quat
