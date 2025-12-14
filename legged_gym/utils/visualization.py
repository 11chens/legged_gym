import torch
import numpy as np
import cv2
import os
from isaacgym import gymapi
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

    def draw_sigma_axes(self, sigma_points, env_idx=0):
        """
        Draw the PCA sigma points as axes in 3D.
        sigma_points: [5, 3] Tensor
        """
        if not self.viewer:
            return
            
        p = sigma_points.cpu().numpy()
        center, head, tail, side1, side2 = p[0], p[1], p[2], p[3], p[4]
        
        # Long Axis (Green)
        self.gym.add_lines(self.viewer, self.env.envs[env_idx], 1, [center[0], center[1], center[2], head[0], head[1], head[2]], [0, 1, 0])
        self.gym.add_lines(self.viewer, self.env.envs[env_idx], 1, [center[0], center[1], center[2], tail[0], tail[1], tail[2]], [0, 1, 0])
        # Short Axis (Red)
        self.gym.add_lines(self.viewer, self.env.envs[env_idx], 1, [center[0], center[1], center[2], side1[0], side1[1], side1[2]], [1, 0, 0])
        self.gym.add_lines(self.viewer, self.env.envs[env_idx], 1, [center[0], center[1], center[2], side2[0], side2[1], side2[2]], [1, 0, 0])
        

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

    def draw_2d_image(self, points_3d_dict, camera_sensor, env_idx=0, filename="debug_cam.png", save_images=False, points_2d_dict=None):
        """
        Get camera image from Isaac Gym, overlay projected points, and save or display.
        points_3d_dict: Dict of {"label": points_tensor_3d}
        points_2d_dict: Dict of {"label": points_tensor_2d} (normalized [0, 1])
        """
        if not hasattr(self.env, 'cam_handles'):
            return

        camera_handle = self.env.cam_handles[env_idx]
        image = self.gym.get_camera_image(self.env.sim, self.env.envs[env_idx], camera_handle, gymapi.IMAGE_COLOR)
        
        h = self.env.cfg.camera_sensor.intrinsics.img_height
        w = self.env.cfg.camera_sensor.intrinsics.img_width
        
        image = image.reshape(h, w, 4)
        image = image[:, :, :3] # RGB
        image = image.astype(np.uint8)
        image = np.ascontiguousarray(image)
        
        colors = {
            "target": (0, 255, 0), # Green
            "sigma": (0, 0, 255),  # Red
            "sigma_2d": (0, 255, 255), # Yellow
            "other": (255, 0, 0)   # Blue
        }
        
        # Draw 3D Points
        if points_3d_dict:
            for label, points in points_3d_dict.items():
                if points is None: continue
                u, v, mask = self.project_points_to_image(points, camera_sensor, env_idx)
                u = u.cpu().numpy()
                v = v.cpu().numpy()
                mask = mask.cpu().numpy()
                color = colors.get(label, (0, 255, 0))
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
                
                color = colors.get(label, (255, 255, 0))
                for i in range(len(pts)):
                    u = int(pts[i, 0] * w)
                    v = int(pts[i, 1] * h)
                    if 0 <= u < w and 0 <= v < h:
                        if i == 0:
                            cv2.circle(image, (u, v), 2, color, 2)
                        else:
                            cv2.drawMarker(image, (u, v), color, markerType=cv2.MARKER_CROSS, markerSize=7, thickness=1)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if save_images:
            path = os.path.join(self.save_dir, filename)
            cv2.imwrite(path, image)
        elif not self.env.headless:
            cv2.imshow("Camera Debug", image)
            cv2.waitKey(1)

