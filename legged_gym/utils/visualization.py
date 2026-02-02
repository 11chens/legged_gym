import torch
import numpy as np
import cv2
import os
from isaacgym import gymapi, gymutil
from isaacgym.torch_utils import quat_apply, quat_rotate_inverse

class VisualizationUtils:
    def __init__(self, env):
        self.env = env
        self.gym = env.gym
        self.viewer = env.viewer
        self.device = env.device
        if hasattr(env, 'log_dir') and env.log_dir is not None:
            self.save_dir = os.path.join(env.log_dir, "vis_frames")
        else:
            self.save_dir = os.path.join(os.getcwd(), "logs", "vis_frames")
        os.makedirs(self.save_dir, exist_ok=True)
        self.frame_idx = 0

        self.sphere_red = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(1, 0, 0))
        self.sphere_green = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(0, 1, 0))
        self.sphere_blue = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(0, 0, 1))

    def draw_sigma_points_3d(self, sigma_points, env_idx=0):
        """
        Draw sigma points in 3D world frame.
        sigma_points: [N, 3] Tensor
        """
        if not self.viewer:
            return
            
        if isinstance(sigma_points, torch.Tensor):
            points = sigma_points.cpu().numpy()
        else:
            points = sigma_points
            
        # Draw Red Crosses for 3D Sigma Points
        d = 0.02 # 2cm size
        color = np.array([1, 0, 0], dtype=np.float32) # Red
        
        for i in range(points.shape[0]):
            px, py, pz = points[i]
            self.gym.add_lines(self.viewer, self.env.envs[env_idx], 1, np.array([px-d, py, pz, px+d, py, pz], dtype=np.float32), color)
            self.gym.add_lines(self.viewer, self.env.envs[env_idx], 1, np.array([px, py-d, pz, px, py+d, pz], dtype=np.float32), color)
            self.gym.add_lines(self.viewer, self.env.envs[env_idx], 1, np.array([px, py, pz-d, px, py, pz+d], dtype=np.float32), color)

    def draw_sigma_axes(self, sigma_points, env_idx=0):
        """
        Draw the PCA sigma points as axes in 3D.
        sigma_points: [N, 3] Tensor
        """
        if not self.viewer:
            return
            
        p = sigma_points.cpu().numpy()
        center = p[0]
        
        # Axis 1 (Green)
        if p.shape[0] >= 3:
            head, tail = p[1], p[2]
            self.gym.add_lines(self.viewer, self.env.envs[env_idx], 1, np.array([center[0], center[1], center[2], head[0], head[1], head[2]], dtype=np.float32), np.array([0, 1, 0], dtype=np.float32))
            self.gym.add_lines(self.viewer, self.env.envs[env_idx], 1, np.array([center[0], center[1], center[2], tail[0], tail[1], tail[2]], dtype=np.float32), np.array([0, 1, 0], dtype=np.float32))
            
        # Axis 2 (Red)
        if p.shape[0] >= 5:
            side1, side2 = p[3], p[4]
            self.gym.add_lines(self.viewer, self.env.envs[env_idx], 1, np.array([center[0], center[1], center[2], side1[0], side1[1], side1[2]], dtype=np.float32), np.array([1, 0, 0], dtype=np.float32))
            self.gym.add_lines(self.viewer, self.env.envs[env_idx], 1, np.array([center[0], center[1], center[2], side2[0], side2[1], side2[2]], dtype=np.float32), np.array([1, 0, 0], dtype=np.float32))
            
        # Axis 3 (Blue)
        if p.shape[0] >= 7:
            top, bottom = p[5], p[6]
            self.gym.add_lines(self.viewer, self.env.envs[env_idx], 1, np.array([center[0], center[1], center[2], top[0], top[1], top[2]], dtype=np.float32), np.array([0, 0, 1], dtype=np.float32))
            self.gym.add_lines(self.viewer, self.env.envs[env_idx], 1, np.array([center[0], center[1], center[2], bottom[0], bottom[1], bottom[2]], dtype=np.float32), np.array([0, 0, 1], dtype=np.float32))
        

    def draw_3d_lines(self, points, color=[0, 1, 0], env_idx=0):
        """
        Draw lines connecting points in 3D.
        points: [N, 3] numpy array or tensor
        """
        if not self.viewer:
            return
            
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
            
        # Draw lines between consecutive points? Or just points?
        # The user's previous code drew axes.
        # Let's just draw small crosses at each point.
        d = 0.01
        for i in range(points.shape[0]):
            px, py, pz = points[i]
            self.gym.add_lines(self.viewer, self.env.envs[env_idx], 1, [px-d, py, pz, px+d, py, pz], color)
            self.gym.add_lines(self.viewer, self.env.envs[env_idx], 1, [px, py-d, pz, px, py+d, pz], color)
            self.gym.add_lines(self.viewer, self.env.envs[env_idx], 1, [px, py, pz-d, px, py, pz+d], color)

    def draw_object_axes(self, object_pos, x_axis, y_axis, z_axis, env_idx=0, axis_len=0.3):
        """
        Draw object coordinate axes.
        object_pos: [3] Tensor
        x_axis, y_axis, z_axis: [3] Tensor
        """
        if not self.viewer:
            return
        
        center = object_pos.cpu().numpy()
        p_x = (object_pos + x_axis * axis_len).cpu().numpy()
        p_y = (object_pos + y_axis * axis_len).cpu().numpy()
        p_z = (object_pos + z_axis * axis_len).cpu().numpy()
        
        # X Axis (Red)
        self.gym.add_lines(self.viewer, self.env.envs[env_idx], 1, np.array([center[0], center[1], center[2], p_x[0], p_x[1], p_x[2]], dtype=np.float32), np.array([1, 0, 0], dtype=np.float32))
        # Y Axis (Green)
        self.gym.add_lines(self.viewer, self.env.envs[env_idx], 1, np.array([center[0], center[1], center[2], p_y[0], p_y[1], p_y[2]], dtype=np.float32), np.array([0, 1, 0], dtype=np.float32))
        # Z Axis (Blue)
        self.gym.add_lines(self.viewer, self.env.envs[env_idx], 1, np.array([center[0], center[1], center[2], p_z[0], p_z[1], p_z[2]], dtype=np.float32), np.array([0, 0, 1], dtype=np.float32))

    def draw_optimal_grasp_pose(self, pos, quat, env_idx=0, axis_len=0.3):
        """
        Draw the optimal grasp pose as a coordinate frame.
        pos: [3] Tensor
        quat: [4] Tensor
        """
        if not self.viewer:
            return

        # Ensure inputs are on the correct device if they are tensors
        if isinstance(pos, torch.Tensor):
            pos = pos.to(self.device)
        if isinstance(quat, torch.Tensor):
            quat = quat.to(self.device)

        x_axis = quat_apply(quat.unsqueeze(0), torch.tensor([[1.0, 0, 0]], device=self.device)).squeeze() * axis_len
        y_axis = quat_apply(quat.unsqueeze(0), torch.tensor([[0, 1.0, 0]], device=self.device)).squeeze() * axis_len
        z_axis = quat_apply(quat.unsqueeze(0), torch.tensor([[0, 0, 1.0]], device=self.device)).squeeze() * axis_len
        
        p0 = pos.cpu().numpy()
        px = (pos + x_axis).cpu().numpy()
        py = (pos + y_axis).cpu().numpy()
        pz = (pos + z_axis).cpu().numpy()
        
        verts = np.stack([p0, px, p0, py, p0, pz], axis=0) # (6, 3)
        colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32) # (3, 3)
        
        self.gym.add_lines(self.viewer, self.env.envs[env_idx], 3, verts, colors)

    def draw_camera_position(self):
        """ Draw camera position in world frame """
        if not self.viewer: return
        
        # Check if camera_world exists
        if not hasattr(self.env, 'camera_world'):
            return

        for i in range(self.env.num_envs):
            pos = self.env.camera_world[i]
            sphere_pose = gymapi.Transform(gymapi.Vec3(pos[0], pos[1], pos[2]), r=None)
            gymutil.draw_lines(self.sphere_red, self.gym, self.viewer, self.env.envs[i], sphere_pose)

    def draw_gripper_position(self, gripper_pos, env_idx=0):
        """ Draw gripper position in world frame """
        if not self.viewer: return
        sphere_pose = gymapi.Transform(gymapi.Vec3(gripper_pos[0], gripper_pos[1], gripper_pos[2]), r=None)
        if self.env.task_flags[env_idx] == 1:  # Grasp task - Red
            gymutil.draw_lines(self.sphere_red, self.gym, self.viewer, self.env.envs[env_idx], sphere_pose)
        else:  # Place task - Green
            gymutil.draw_lines(self.sphere_green, self.gym, self.viewer, self.env.envs[env_idx], sphere_pose)

    def draw_fov(self, env_idx=0):
        """ Draw FOV lines for debugging """
        if not self.viewer: return

        cam = self.env.camera_sensor
        # Use cached params if available, else from sensor
        # Fallback
        fx = cam.fx
        fy = cam.fy
        cx = cam.cx
        cy = cam.cy
        img_w = cam.img_width
        img_h = cam.img_height

        # Define corners in image pixel coordinates
        corners_pix = torch.tensor([
            [0, 0],
            [img_w, 0],
            [img_w, img_h],
            [0, img_h]
        ], device=self.device, dtype=torch.float)
        
        # Transform to Camera Frame
        depth = 3.5 # Max visual distance
        
        # x = (u - cx) * Z / fx, y = (v - cy) * Z / fy, z = Z
        corners_cam_x = (corners_pix[:, 0] - cx) * depth / fx
        corners_cam_y = (corners_pix[:, 1] - cy) * depth / fy
        corners_cam_z = torch.full((4,), depth, device=self.device)
        
        corners_cam = torch.stack([corners_cam_x, corners_cam_y, corners_cam_z], dim=-1) # (4, 3)
        
        # Transform to World Frame directly
        # P_world = P_cam @ R_world_to_cam + T_world
        R_w2c = self.env.R_world_to_cam[env_idx] # [3, 3]
        T_w = self.env.camera_world[env_idx] # [3]
        
        corners_world = torch.matmul(corners_cam, R_w2c) + T_w
        center_world = T_w
        
        # Draw lines
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 0, 1))
        
        corners = corners_world.cpu().numpy()
        center = center_world.cpu().numpy()
        
        def draw_thick_line(start, end, num_spheres=50):
            for k in range(num_spheres + 1):
                t = k / num_spheres
                pos = start + (end - start) * t
                pose = gymapi.Transform(gymapi.Vec3(pos[0], pos[1], pos[2]), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, None, pose)

        # 4 lines from center to corners
        for k in range(4):
            draw_thick_line(center, corners[k], num_spheres=50)
            
        # 4 lines connecting corners
        for k in range(4):
            draw_thick_line(corners[k], corners[(k+1)%4], num_spheres=50)


    def draw_hint_pose(self, pos, quat, env_idx=0, axis_len=0.15):
        """
        Draw the hint pose as a coordinate frame.
        pos: [3] Tensor
        quat: [4] Tensor
        """
        self.draw_optimal_grasp_pose(pos, quat, env_idx, axis_len)

    def draw_start_pos(self, pos, env_idx=0):
        """
        Draw the start position marker.
        pos: [3] Tensor
        """
        if not self.viewer:
            return
            
        if isinstance(pos, torch.Tensor):
            pos = pos.cpu().numpy()
            
        d = 0.05 # 5cm marker size
        
        # Yellow
        color = np.array([1, 1, 0], dtype=np.float32)
        px, py, pz = pos
        self.gym.add_lines(self.viewer, self.env.envs[env_idx], 1, np.array([px-d, py, pz, px+d, py, pz], dtype=np.float32), color)
        self.gym.add_lines(self.viewer, self.env.envs[env_idx], 1, np.array([px, py-d, pz, px, py+d, pz], dtype=np.float32), color)
        self.gym.add_lines(self.viewer, self.env.envs[env_idx], 1, np.array([px, py, pz-d, px, py, pz+d], dtype=np.float32), color)

    def draw_sequential_reaching_debug(self, hint_pos, optimal_pos, gripper_pos, env_idx=0):
        """
        Draw debug info for sequential reaching reward.
        hint_pos: [3] Tensor
        optimal_pos: [3] Tensor
        gripper_pos: [3] Tensor
        """
        if not self.viewer:
            return
            
        # Ensure inputs are on CPU numpy
        if isinstance(hint_pos, torch.Tensor): hint_pos = hint_pos.cpu().numpy()
        if isinstance(optimal_pos, torch.Tensor): optimal_pos = optimal_pos.cpu().numpy()
        if isinstance(gripper_pos, torch.Tensor): gripper_pos = gripper_pos.cpu().numpy()
        
        # 1. Draw Path (Hint -> Optimal) - Cyan Line
        self.gym.add_lines(self.viewer, self.env.envs[env_idx], 1, 
                           np.array([hint_pos[0], hint_pos[1], hint_pos[2], 
                                     optimal_pos[0], optimal_pos[1], optimal_pos[2]], dtype=np.float32), 
                           np.array([0, 1, 1], dtype=np.float32))
        
        # 2. Calculate Closest Point (2D Logic to match reward)
        # We only consider X, Y for the projection, ignoring Z.
        v_path = optimal_pos - hint_pos
        v_path_2d = v_path[:2]
        len_sq_2d = np.sum(v_path_2d**2)
        
        v_gripper = gripper_pos - hint_pos
        v_gripper_2d = v_gripper[:2]
        
        t = np.sum(v_gripper_2d * v_path_2d) / (len_sq_2d + 1e-6)
        t_clamped = np.clip(t, 0.0, 1.0)
        
        # Point on the 3D line
        p_closest = hint_pos + t_clamped * v_path
        
        # 3. Draw Closest Point - Magenta Cross
        d = 0.03
        px, py, pz = p_closest
        color_closest = np.array([1, 0, 1], dtype=np.float32)
        self.gym.add_lines(self.viewer, self.env.envs[env_idx], 1, np.array([px-d, py, pz, px+d, py, pz], dtype=np.float32), color_closest)
        self.gym.add_lines(self.viewer, self.env.envs[env_idx], 1, np.array([px, py-d, pz, px, py+d, pz], dtype=np.float32), color_closest)
        self.gym.add_lines(self.viewer, self.env.envs[env_idx], 1, np.array([px, py, pz-d, px, py, pz+d], dtype=np.float32), color_closest)
        
        # 4. Draw Error Line (Gripper -> Closest) - Red Line
        self.gym.add_lines(self.viewer, self.env.envs[env_idx], 1,
                           np.array([gripper_pos[0], gripper_pos[1], gripper_pos[2],
                                     p_closest[0], p_closest[1], p_closest[2]], dtype=np.float32),
                           np.array([1, 0, 0], dtype=np.float32))

    def draw_target_points(self, local_points, object_pos, object_quat, num_vis_points, env_idx=0, weights=None):
        """
        Downsample and draw target points in world frame.
        local_points: [N, 3] Tensor (all points)
        object_pos: [3] Tensor
        object_quat: [4] Tensor
        num_vis_points: int
        weights: [N, 1] visibility weights
        """
        if not self.viewer:
            return
            
        # 1. Draw "Ghost" Body (All points in Blue)
        num_all = local_points.shape[0]
        stride_all = max(1, int(num_all // num_vis_points))
        pts_all_local = local_points[::stride_all]
        pts_all_world = quat_apply(
            object_quat.unsqueeze(0).expand(pts_all_local.shape[0], -1), 
            pts_all_local
        ) + object_pos
        self.draw_3d_lines(pts_all_world, color=[0, 0, 0.8], env_idx=env_idx) # Blue Ghost

        # 2. Draw "Visible" Points (Filtered in Green)
        if weights is not None:
            mask = weights.squeeze() > 1e-6
            local_points_vis = local_points[mask]
        else:
            local_points_vis = local_points

        num_vis = local_points_vis.shape[0]
        if num_vis > 0:
            stride_vis = max(1, int(num_vis // num_vis_points))
            pts_vis_local = local_points_vis[::stride_vis]
            pts_vis_world = quat_apply(
                object_quat.unsqueeze(0).expand(pts_vis_local.shape[0], -1), 
                pts_vis_local
            ) + object_pos
            self.draw_3d_lines(pts_vis_world, color=[0, 1, 0], env_idx=env_idx) # Bright Green
            return pts_vis_world
        
        return None

    def draw_head_tail_points(self, head_pos, tail_pos, env_idx=0):
        """
        Draw markers (crosses) at head and tail points.
        """
        if not self.viewer:
            return
            
        head = head_pos.cpu().numpy()
        tail = tail_pos.cpu().numpy()
        
        d = 0.05 # 5cm marker size
        
        # Head (Cyan)
        color_head = np.array([0, 1, 1], dtype=np.float32)
        px, py, pz = head
        self.gym.add_lines(self.viewer, self.env.envs[env_idx], 1, np.array([px-d, py, pz, px+d, py, pz], dtype=np.float32), color_head)
        self.gym.add_lines(self.viewer, self.env.envs[env_idx], 1, np.array([px, py-d, pz, px, py+d, pz], dtype=np.float32), color_head)
        self.gym.add_lines(self.viewer, self.env.envs[env_idx], 1, np.array([px, py, pz-d, px, py, pz+d], dtype=np.float32), color_head)

        # Tail (Magenta)
        color_tail = np.array([1, 0, 1], dtype=np.float32)
        px, py, pz = tail
        self.gym.add_lines(self.viewer, self.env.envs[env_idx], 1, np.array([px-d, py, pz, px+d, py, pz], dtype=np.float32), color_tail)
        self.gym.add_lines(self.viewer, self.env.envs[env_idx], 1, np.array([px, py-d, pz, px, py+d, pz], dtype=np.float32), color_tail)
        self.gym.add_lines(self.viewer, self.env.envs[env_idx], 1, np.array([px, py, pz-d, px, py, pz+d], dtype=np.float32), color_tail)

    def project_points_to_image(self, points_3d, camera_sensor, env_idx=0):
        """
        Project 3D points (in World Frame) to Image Plane.
        points_3d: [N, 3] Tensor in World Frame
        """
        # 1. World -> Base
        # Use pre-calculated base_quat and root_states from env
        base_quat = self.env.base_quat[env_idx]
        base_pos = self.env.root_states[env_idx, :3]
        
        points_base = quat_rotate_inverse(base_quat.unsqueeze(0).expand(len(points_3d), -1), points_3d - base_pos)
        
        # 2. Base -> Image
        # Note: We cannot use camera_sensor.transform() directly because it expects batch_size=num_envs
        # We manually apply the transform using the specific env's parameters
        R = camera_sensor.R[env_idx]
        T = camera_sensor.T[env_idx]
        
        # Base -> Camera: P_cam = R @ (P_base - T)
        points_cam = (R @ (points_base - T).T).T
        
        # Camera -> Image
        def get_val(param, idx):
            if isinstance(param, torch.Tensor) and param.dim() > 0:
                return param[idx]
            return param

        fx = get_val(camera_sensor.fx, env_idx)
        fy = get_val(camera_sensor.fy, env_idx)
        cx = get_val(camera_sensor.cx, env_idx)
        cy = get_val(camera_sensor.cy, env_idx)
        
        x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
        
        u = (x / z) * fx + cx
        v = (y / z) * fy + cy
        
        return u, v, (z > 0.1)

    def draw_2d_image(self, points_3d_dict, camera_sensor, env_idx=0, filename="debug_cam.png", save_images=False, points_2d_dict=None, lines_3d_dict=None):
        """
        Get camera image from Isaac Gym, overlay projected points, and save or display.
        points_3d_dict: Dict of {"label": points_tensor_3d}
        points_2d_dict: Dict of {"label": points_tensor_2d} (normalized [0, 1])
        lines_3d_dict: Dict of {"label": points_tensor_3d (2, 3)} - Start and End points of lines
        """
        # Get dimensions from config
        h = camera_sensor.cfg.intrinsics.img_height
        w = camera_sensor.cfg.intrinsics.img_width
        
        # Try to get image from Isaac Gym if available
        # if hasattr(self.env, 'cam_handles') and self.env.cam_handles and len(self.env.cam_handles) > env_idx:
        if self.env.enable_camera:
            camera_handle = self.env.cam_handles[env_idx]
            image = self.gym.get_camera_image(self.env.sim, self.env.envs[env_idx], camera_handle, gymapi.IMAGE_COLOR)
            image = image.reshape(h, w, 4)
            image = image[:, :, :3] # RGB
            image = image.astype(np.uint8)
            image = np.ascontiguousarray(image)
        
        # If no image (camera disabled), create a blank one
        else:
            image = np.zeros((h, w, 3), dtype=np.uint8)
        
        colors = {
            "target": (0, 255, 0), # Green
            "sigma_3d": (255, 0, 0),  # Red
            "sigma_2d": (255, 255, 255), # White
            "y_spread": (0, 255, 255), # Yellow
            "other": (0, 0, 255)   # Blue
        }
        
        # Draw 3D Points
        if points_3d_dict:
            for label, points in points_3d_dict.items():
                if points is None: continue
                u, v, mask = self.project_points_to_image(points, camera_sensor, env_idx)
                u = u.cpu().numpy()
                v = v.cpu().numpy()
                mask = mask.cpu().numpy()
                color = colors.get(label, (0, 255, 0)) # Default Green
                for i in range(len(u)):
                    if mask[i]:
                        cv2.circle(image, (int(u[i]), int(v[i])), 3, color, -1)

        # Draw 2D Points
        if points_2d_dict:
            for label, points in points_2d_dict.items():
                if points is None: continue
                if isinstance(points, torch.Tensor):
                    pts = points.cpu().numpy()
                else:
                    pts = points
                
                color = colors.get(label, (255, 255, 0)) # Default Yellow
                for i in range(len(pts)):
                    u = int(pts[i, 0] * w)
                    v = int(pts[i, 1] * h)
                    if 0 <= u < w and 0 <= v < h:
                        if i == 0:
                            cv2.circle(image, (u, v), 2, color, 2)
                        else:
                            cv2.drawMarker(image, (u, v), color, markerType=cv2.MARKER_CROSS, markerSize=6, thickness=2)

        # Draw 3D Lines
        if lines_3d_dict:
            for label, points in lines_3d_dict.items():
                if points is None: continue
                # points should be [2, 3] (start, end)
                u, v, mask = self.project_points_to_image(points, camera_sensor, env_idx)
                u = u.cpu().numpy()
                v = v.cpu().numpy()
                mask = mask.cpu().numpy()
                color = colors.get(label, (0, 255, 255))
                if mask[0] and mask[1]:
                    p1 = (int(u[0]), int(v[0]))
                    p2 = (int(u[1]), int(v[1]))
                    cv2.line(image, p1, p2, color, 2)

        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if save_images:
            path = os.path.join(self.save_dir, filename)
            cv2.imwrite(path, image_bgr)
        elif not self.env.headless:
            cv2.imshow("Camera Debug", image_bgr)
            cv2.waitKey(1)

        return image_bgr

