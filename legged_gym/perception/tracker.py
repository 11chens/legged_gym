import torch
import time
from .surface_geometry import SurfaceShape, Sphere, Cuboid, Cylinder
from .perception_utils import compute_visibility_weights, project_points, compute_weighted_pca, generate_sigma_points
from isaacgym.torch_utils import quat_apply

class PCATargetTracker:
    def __init__(self, shape_type: str, shape_params: torch.Tensor, num_envs: int, device: str, num_sample_points: int = 1000, local_points=None, local_normals=None):
        """
        Initializes the PCA Target Tracker.
        
        Args:
            shape_type (str): 'sphere', 'cuboid', 'cylinder', or 'mixed'.
            shape_params (Tensor): Shape parameters.
            num_envs (int): Number of environments.
            device (str): Device to run on.
            num_sample_points (int): Number of points to sample on the surface.
            local_points (Tensor, optional): Pre-sampled local points [num_envs, num_points, 3].
            local_normals (Tensor, optional): Pre-sampled local normals [num_envs, num_points, 3].
        """
        self.device = device
        self.num_envs = num_envs
        self.num_points = num_sample_points
        
        if local_points is not None and local_normals is not None:
            self.local_points = local_points
            self.local_normals = local_normals
        else:
            # Initialize shape and pre-sample local points
            if shape_type == 'sphere':
                self.shape = Sphere(device)
            elif shape_type == 'cuboid':
                self.shape = Cuboid(device)
            elif shape_type == 'cylinder':
                self.shape = Cylinder(device)
            else:
                raise ValueError(f"Unknown shape type: {shape_type}")
                
            # Sample once and cache
            # shape_params: [num_envs, num_params] or [1, num_params] if shared
            if shape_params.shape[0] == 1 and num_envs > 1:
                shape_params = shape_params.repeat(num_envs, 1)
                
            # These are in the object's local frame (centered at 0,0,0, aligned with axes)
            self.local_points, self.local_normals = self.shape.sample_surface(self.num_points, self.num_envs, shape_params)
            # self.local_points: [num_envs, num_points, 3]
        
        self.debug_counter = 0

    def compute_features(self, object_pos, object_quat, camera_params, camera_transform, use_geometric_weight=True, debug_timer=False, debug_info=False):
        """
        Computes PCA features for the tracked object.
        
        Args:
            object_pos (Tensor): [num_envs, 3] World position of the object.
            object_quat (Tensor): [num_envs, 4] World orientation (x, y, z, w).
            camera_params (dict): Camera intrinsics.
            camera_transform (dict): Camera extrinsics ('R', 'T').
            use_geometric_weight (bool): Whether to use geometric weighting for anti-drift.
            debug_timer (bool): If True, prints timing info.
            debug_info (bool): If True, prints detailed debug information.
            
        Returns:
            sigma_points (Tensor): [num_envs, 5, 2]
            mean (Tensor): [num_envs, 2]
            eigvals (Tensor): [num_envs, 2]
            eigvecs (Tensor): [num_envs, 2, 2]
            weights (Tensor): [num_envs, num_points, 1] Visibility weights.
            points_2d (Tensor): [num_envs, num_points, 2] Projected points.
        """
        t0 = time.time()
        
        # 1. Transform to World Frame
        # Expand quat for broadcasting: [num_envs, 1, 4]
        quat_expanded = object_quat.unsqueeze(1).expand(-1, self.num_points, -1)
        
        # Apply rotation to points and normals
        # quat_apply expects (x, y, z, w)
        points_rot = quat_apply(quat_expanded, self.local_points)
        normals_rot = quat_apply(quat_expanded, self.local_normals)
        
        # Add translation
        points_world = points_rot + object_pos.unsqueeze(1)
        normals_world = normals_rot
        
        t1 = time.time()
        
        # 2. Visibility
        cam_pos = camera_transform['T'] # [num_envs, 3]
        weights = compute_visibility_weights(points_world, normals_world, cam_pos, use_geometric_weight=use_geometric_weight)
        
        t2 = time.time()
        
        # 3. Projection
        points_2d, valid_mask = project_points(points_world, camera_params, camera_transform)
        
        # Filter weights by valid mask (Z > 0)
        weights = weights * valid_mask.unsqueeze(-1).float()
        
        t3 = time.time()
        
        # 4. PCA
        mean, eigvals, eigvecs, valid_2d = compute_weighted_pca(points_2d, weights)
        
        t4 = time.time()
        
        # 5. Sigma Points
        sigma_points = generate_sigma_points(mean, eigvals, eigvecs, alpha=1.5)
        
        # 6. PCA (3D)
        mean_3d, eigvals_3d, eigvecs_3d, valid_3d = compute_weighted_pca(points_world, weights)
        sigma_points_3d = generate_sigma_points(mean_3d, eigvals_3d, eigvecs_3d, alpha=1.5)
        
        # Combine validity
        is_valid = valid_2d & valid_3d
        
        t5 = time.time()
        
        if debug_info:
            self.debug_counter += 1
            if debug_timer or (self.debug_counter % 50 == 0):
                print(f"\n[Tracker Debug Env 0] Step {self.debug_counter}")
                print(f"  Cam Pos: {cam_pos[0].tolist()}")
                print(f"  Obj Pos: {object_pos[0].tolist()}")
                print(f"  Valid: {is_valid[0].item()}")
                
                # Check if inputs are changing
                if not hasattr(self, '_debug_last_cam_pos'):
                    self._debug_last_cam_pos = cam_pos[0].clone()
                    self._debug_last_pts_2d = points_2d[0].clone()
                else:
                    cam_diff = (cam_pos[0] - self._debug_last_cam_pos).norm().item()
                    pts_diff = (points_2d[0] - self._debug_last_pts_2d).norm().item()
                    print(f"  Delta Cam Pos: {cam_diff:.6f}")
                    print(f"  Delta Pts 2D:  {pts_diff:.6f}")
                    
                    if cam_diff > 1e-4 and pts_diff < 1e-5:
                        print(f"  WARNING: Camera moved but Points 2D didn't change!")
                    
                    self._debug_last_cam_pos = cam_pos[0].clone()
                    self._debug_last_pts_2d = points_2d[0].clone()
                    
                vis_mask = weights[0, :, 0] > 0
                print(f"  Visible Points: {vis_mask.sum().item()} / {self.num_points}")
                if vis_mask.sum() > 0:
                    print(f"  Mean 2D: {points_2d[0][vis_mask].mean(dim=0).tolist()}")

        if debug_timer:
            print(f"PCA Tracker Timing:")
            print(f"  Transform: {(t1-t0)*1000:.3f} ms")
            print(f"  Visibility: {(t2-t1)*1000:.3f} ms")
            print(f"  Projection: {(t3-t2)*1000:.3f} ms")
            print(f"  PCA (2D):   {(t4-t3)*1000:.3f} ms")
            print(f"  PCA (3D):   {(t5-t4)*1000:.3f} ms")
            print(f"  Total:      {(t5-t0)*1000:.3f} ms")
        
        return sigma_points, mean, eigvals, eigvecs, weights, points_2d, sigma_points_3d, is_valid
