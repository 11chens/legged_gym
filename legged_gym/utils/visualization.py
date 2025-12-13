import torch
import numpy as np
import cv2
import os
from isaacgym import gymapi
from isaacgym.torch_utils import quat_apply

class VisualizationUtils:
    def __init__(self, env):
        self.env = env
        self.gym = env.gym
        self.viewer = env.viewer
        self.device = env.device
        if hasattr(env, 'log_dir'):
            self.save_dir = os.path.join(env.log_dir, "vis_frames")
        else:
            self.save_dir = os.path.join(os.getcwd(), "logs", "vis_frames")
        os.makedirs(self.save_dir, exist_ok=True)
        self.frame_idx = 0

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

    def project_points_to_image(self, points_3d, camera_sensor, env_idx=0):
        """
        Project 3D points (in World Frame) to Image Plane.
        points_3d: [N, 3] Tensor in World Frame
        camera_sensor: CameraSensor object
        """
        # 1. Transform World -> Camera
        # P_cam = R * (P_world - T_world)
        # Note: CameraSensor stores R (Base->Cam) and T (Cam in Base).
        # We need World->Cam.
        # P_base = R_base_world * (P_world - T_base_world)
        # P_cam = R_cam_base * (P_base - T_cam_base)
        
        # Actually, let's look at how CameraSensor works or how we did it in legged_robot_nav.
        # In legged_robot_nav, we had sigma_points_body (in Base Frame).
        # And we used camera_sensor.R (Base->Cam) and T (Cam in Base).
        
        # So first, World -> Base
        root_state = self.env.root_states[env_idx]
        base_pos = root_state[:3]
        base_quat = root_state[3:7]
        
        points_3d_env = points_3d # [N, 3]
        
        # P_body = R_world_base * (P_world - P_base)
        # quat_rotate_inverse rotates by conjugate.
        # base_quat is World->Base rotation? No, usually Base->World (orientation).
        # So quat_rotate_inverse does World->Base.
        
        delta = points_3d_env - base_pos
        # Expand quat for broadcasting
        quat_expanded = base_quat.unsqueeze(0).expand(points_3d.shape[0], -1)
        # If quat is Base->World, quat_apply is Base->World.
        # We want World->Base. So we need inverse quat.
        # quat_rotate_inverse is what we want.
        from isaacgym.torch_utils import quat_rotate_inverse
        points_body = quat_rotate_inverse(quat_expanded, delta)
        
        # 2. Base -> Camera
        # P_cam = R * (P_body - T)
        # R: [batch, 3, 3], T: [batch, 3]
        R = camera_sensor.R[env_idx] # [3, 3]
        T = camera_sensor.T[env_idx] # [3]
        
        delta_cam = points_body - T
        # R @ delta_cam^T
        points_cam = torch.matmul(R, delta_cam.T).T # [N, 3]
        
        # 3. Camera -> Image
        x = points_cam[:, 0]
        y = points_cam[:, 1]
        z = points_cam[:, 2]
        
        def get_val(param, idx):
            if isinstance(param, torch.Tensor):
                if param.dim() > 0:
                    return param[idx]
                else:
                    return param
            return param

        fx = get_val(camera_sensor.fx, env_idx)
        fy = get_val(camera_sensor.fy, env_idx)
        cx = get_val(camera_sensor.cx, env_idx)
        cy = get_val(camera_sensor.cy, env_idx)
        
        # Avoid div by zero (behind camera)
        mask = z > 0.1
        
        u = (x / z) * fx + cx
        v = (y / z) * fy + cy
        
        return u, v, mask

    def save_camera_debug_image(self, points_3d_dict, camera_sensor, env_idx=0, filename="debug_cam.png"):
        """
        Get camera image from Isaac Gym, overlay projected points, and save.
        points_3d_dict: Dict of {"label": points_tensor}
        """
        # 1. Get Image from Isaac Gym
        # We need to access the camera handle.
        # In LeggedRobot, camera handles are usually stored.
        # Let's assume self.env.cam_handles exists.
        
        if not hasattr(self.env, 'cam_handles'):
            print("No camera handles found.")
            return

        camera_handle = self.env.cam_handles[env_idx]
        
        # Retrieve image
        # gym.get_camera_image returns numpy array
        image = self.gym.get_camera_image(self.env.sim, self.env.envs[env_idx], camera_handle, gymapi.IMAGE_COLOR)
        
        # Reshape: [H, W, 4] (RGBA)
        def get_val(param, idx):
            if isinstance(param, torch.Tensor):
                if param.dim() > 0:
                    return param[idx].item()
                else:
                    return param.item()
            return param

        h = int(get_val(camera_sensor.img_height, env_idx))
        w = int(get_val(camera_sensor.img_width, env_idx))
        image = image.reshape(h, w, 4)
        image = image[:, :, :3] # RGB
        image = image.astype(np.uint8)
        image = np.ascontiguousarray(image) # cv2 needs contiguous

    def save_camera_debug_image(self, points_3d_dict, camera_sensor, env_idx=0, filename="debug_cam.png"):
        """
        Get camera image from Isaac Gym, overlay projected points, and save.
        points_3d_dict: Dict of {"label": points_tensor}
        """
        # 1. Get Image from Isaac Gym
        # We need to access the camera handle.
        # In LeggedRobot, camera handles are usually stored.
        # Let's assume self.env.camera_handles exists.
        
        if not hasattr(self.env, 'cam_handles'):
            print("No camera handles found.")
            return

        camera_handle = self.env.cam_handles[env_idx]
        
        # Retrieve image
        # gym.get_camera_image returns numpy array
        image = self.gym.get_camera_image(self.env.sim, self.env.envs[env_idx], camera_handle, gymapi.IMAGE_COLOR)
        
        # Reshape: [H, W, 4] (RGBA)
        # Use the fixed config dimensions used to create the camera, 
        # because CameraSensor might have randomized dimensions.
        h = self.env.cfg.camera_sensor.intrinsics.img_height
        w = self.env.cfg.camera_sensor.intrinsics.img_width
        
        image = image.reshape(h, w, 4)
        image = image[:, :, :3] # RGB
        image = image.astype(np.uint8)
        image = np.ascontiguousarray(image) # cv2 needs contiguous
        
        # 2. Project and Draw Points
        colors = {
            "target": (0, 255, 0), # Green
            "sigma": (0, 0, 255),  # Red
            "other": (255, 0, 0)   # Blue
        }
        
        for label, points in points_3d_dict.items():
            if points is None: continue
            
            u, v, mask = self.project_points_to_image(points, camera_sensor, env_idx)
            
            u = u.cpu().numpy()
            v = v.cpu().numpy()
            mask = mask.cpu().numpy()
            
            color = colors.get(label, (255, 255, 255))
            
            for i in range(len(u)):
                if mask[i]:
                    cv2.circle(image, (int(u[i]), int(v[i])), 3, color, -1)
        
        # 3. Save
        # Convert RGB to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        path = os.path.join(self.save_dir, filename)
        cv2.imwrite(path, image)
        # print(f"Saved debug image to {path}")

