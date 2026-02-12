import time
from .surface_geometry import SurfaceShape, Sphere, Cuboid, Cylinder, Box, Ellipsoid
from .perception_utils import compute_visibility_weights, compute_occlusion_mask, project_points, compute_weighted_pca, generate_sigma_points, generate_structured_noisy_sigma_points
import torch
from isaacgym.torch_utils import quat_apply, quat_rotate_inverse

class PCATargetTracker:
    def __init__(self, shape_type: str, shape_params: torch.Tensor, num_envs: int, device: str, num_sample_points: int = 1000, local_points=None, local_normals=None, object_dims=None):
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
            object_dims (Tensor, optional): [num_envs, 3] Object dimensions (L, W, H).
        """
        self.device = device
        self.num_envs = num_envs
        self.num_points = num_sample_points
        self.alpha = 2.0  # Scaling factor for sigma points
        
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
            elif shape_type == 'box':
                self.shape = Box(device)
            elif shape_type == 'ellipsoid':
                self.shape = Ellipsoid(device)
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

        if object_dims is not None:
            self.object_dims = object_dims
        else:
            # Calculate from shape_params
            if shape_params.shape[0] == 1 and num_envs > 1:
                shape_params_expanded = shape_params.repeat(num_envs, 1)
            else:
                shape_params_expanded = shape_params

            self.object_dims = self.compute_object_dims(shape_type, shape_params_expanded, device)

    @staticmethod
    def compute_object_dims(shape_type, shape_params, device):
        """
        Compute object dimensions (L, W, H) from shape parameters.
        Args:
            shape_type (str): 'sphere', 'cuboid', 'cylinder'
            shape_params (Tensor): [N, K]
            device (str): Device
        Returns:
            dims (Tensor): [N, 3]
        """
        num_envs = shape_params.shape[0]
        dims = torch.zeros(num_envs, 3, device=device)
        
        if shape_type == 'sphere':
            # shape_params: [N, 1] (radius)
            d = 2 * shape_params[:, 0]
            dims[:, 0] = d
            dims[:, 1] = d
            dims[:, 2] = d
        elif shape_type == 'cuboid' or shape_type == 'box':
            # shape_params: [N, 3] (dims)
            dims[:] = shape_params
        elif shape_type == 'cylinder' or shape_type == 'cylinder_well':
            # shape_params: [N, 2] (radius, height)
            d = 2 * shape_params[:, 0]
            h = shape_params[:, 1]
            dims[:, 0] = d
            dims[:, 1] = d
            dims[:, 2] = h
        elif shape_type == 'ellipsoid':
            # shape_params: [N, 3] (dims: diameter_x, diameter_y, diameter_z)
            dims[:] = shape_params
        return dims
    

    def transform_points_to_world(self, object_pos, object_quat):
        """
        Transforms local points to world frame.
        
        Args:
            object_pos (Tensor): [num_envs, 3] World position of the object.
            object_quat (Tensor): [num_envs, 4] World orientation (x, y, z, w).
            
        Returns:
            points_world (Tensor): [num_envs, num_points, 3]
            normals_world (Tensor): [num_envs, num_points, 3]
        """
        # Expand quat for broadcasting: [num_envs, 1, 4]
        quat_expanded = object_quat.unsqueeze(1).expand(-1, self.num_points, -1)
        
        # Apply rotation to points and normals
        # quat_apply expects (x, y, z, w)
        points_rot = quat_apply(quat_expanded, self.local_points)
        normals_rot = quat_apply(quat_expanded, self.local_normals)
        
        # Add translation
        points_world = points_rot + object_pos.unsqueeze(1)
        normals_world = normals_rot
        
        return points_world, normals_world


    def compute_features(self, object_pos, object_quat, camera_params, camera_transform, use_geometric_weight=True, noise_params=None, pre_pca_noise_params=None, debug_timer=False, debug_info=False, alpha=None, noise_active_mask=None):
        """
        Computes PCA features for the tracked object.
        
        Args:
            object_pos (Tensor): [num_envs, 3] World position of the object.
            object_quat (Tensor): [num_envs, 4] World orientation (x, y, z, w).
            camera_params (dict): Camera intrinsics.
            camera_transform (dict): Camera extrinsics ('R', 'T').
            use_geometric_weight (bool): Whether to use geometric weighting for anti-drift.
            noise_params (dict, optional): Dict containing structured noise scales (post-PCA).
            pre_pca_noise_params (dict, optional): Dict containing noise parameters for raw points (pre-PCA).
            debug_timer (bool): If True, prints timing info.
            debug_info (bool): If True, prints detailed debug information.
            alpha (Tensor, optional): [num_envs, 1] Scaling factor for sigma points.
            noise_active_mask (Tensor, optional): [num_envs] Boolean mask. If provided, pre-PCA noise is only applied where True.
            
        Returns:
            sigma_points_2d (Tensor): [num_envs, 5, 2]
            mean_2d (Tensor): [num_envs, 2]
            eigvals_2d (Tensor): [num_envs, 2]
            eigvecs_2d (Tensor): [num_envs, 2, 2]
            weights (Tensor): [num_envs, num_points, 1] Visibility weights.
            points_2d (Tensor): [num_envs, num_points, 2] Projected points.
            sigma_points_3d (Tensor): [num_envs, 7, 3]
            is_valid (Tensor): [num_envs] Validity flag.
        """
        if alpha is None:
            alpha = self.alpha

        t0 = time.time()
        
        # 1. Transform to World Frame: points_world, normals_world: [num_envs, num_points, 3]
        points_world, normals_world = self.transform_points_to_world(object_pos, object_quat)
        
        t1 = time.time()
        
        # 2. Projection: points_2d: [num_envs, num_points, 2], valid_mask: [num_envs, num_points]
        # Valid mask already includes Z > 0 and inside FOV
        points_2d, valid_mask = project_points(points_world, camera_params, camera_transform)
        
        t2 = time.time()
        
        # 3. Visibility: weights: [num_envs, num_points, 1]
        cam_pos = camera_transform['T'] # [num_envs, 3] in world frame
        
        # [BLOCKING DEPTH BUFFER]
        # Build a Z-buffer using ALL points in FOV to handle self-occlusion.
        # grid_res=64 combined with 5000 points provides a dense mask without "narrowing" faces.
        R_cam = camera_transform['R']
        delta = points_world - cam_pos.unsqueeze(1)
        depths = torch.matmul(R_cam.unsqueeze(1), delta.unsqueeze(-1)).squeeze(-1)[..., 2]
        
        occlusion_mask = compute_occlusion_mask(points_2d, depths, valid_mask, grid_res=64)
        
        # Local Visibility (Back-face culling)
        # ONLY AFTER blocking do we apply the normal-based culling
        weights = compute_visibility_weights(points_world, normals_world, cam_pos, use_geometric_weight=use_geometric_weight)
        
        # Combine everything
        weights = weights * occlusion_mask * valid_mask.unsqueeze(-1).float()
        
        t3 = time.time()
        
        # [PRE-PCA NOISE INJECTION]
        # Simulate imperfect perception (segmentation noise, depth noise, outliers)
        # Note: We must apply noise to points that are used for PCA.
        # 2D case: points_2d
        # 3D case: points_camera (computed later)
        
        # For 2D PCA, we act on points_2d.
        points_2d_noisy = points_2d.clone()
        
        if pre_pca_noise_params is not None:
             # Prepare mask broadcasting
             mask_broad = 1.0
             if noise_active_mask is not None:
                 mask_broad = noise_active_mask.view(-1, 1, 1).float()

             # 1. Pixel Noise (2D)
             if 'pixel_std' in pre_pca_noise_params and pre_pca_noise_params['pixel_std'] > 0:
                 noise = torch.randn_like(points_2d) * pre_pca_noise_params['pixel_std']
                 points_2d_noisy += noise * mask_broad
                 
             # 2. Outliers (2D) - random points in [0,1]
             if 'outlier_prob' in pre_pca_noise_params and pre_pca_noise_params['outlier_prob'] > 0:
                 prob = pre_pca_noise_params['outlier_prob']
                 mask = torch.rand(points_2d.shape[:2], device=self.device) < prob
                 
                 # Only apply outliers where noise is active
                 if noise_active_mask is not None:
                      mask = mask & noise_active_mask.unsqueeze(-1)

                 # Replace masked points with random uniform noise
                 outliers = torch.rand((mask.sum(), 2), device=self.device)
                 points_2d_noisy[mask] = outliers

        
        # 4. PCA (2D): mean_2d: [num_envs, 2], eigvals_2d: [num_envs, 2], eigvecs_2d: [num_envs, 2, 2], valid_2d: [num_envs]
        # Use noisy points for PCA inputs
        mean_2d, eigvals_2d, eigvecs_2d, valid_2d = compute_weighted_pca(points_2d_noisy, weights)
        
        # 5. Generate 2D Sigma Points: sigma_points_2d: [num_envs, 5, 2] in Image Plane

        if noise_params is not None:
            sigma_points_2d = generate_structured_noisy_sigma_points(
                mean_2d, eigvals_2d, eigvecs_2d, 
                pos_noise=noise_params.get('pos_2d', 0),
                scale_noise=noise_params.get('scale_2d', 0),
                rot_noise=noise_params.get('rot_2d', 0),
                alpha=alpha
            )
        else:
            sigma_points_2d = generate_sigma_points(mean_2d, eigvals_2d, eigvecs_2d, alpha=alpha)

        t4 = time.time()
        
        # 6. PCA (3D): Performed in CAMERA FRAME to apply anisotropic depth noise correctly
        R_cam = camera_transform['R'] # [N, 3, 3] World to Camera
        T_cam = camera_transform['T'] # [N, 3] Camera position in World frame
        
        # Transform points_world to Camera Frame
        # P_cam = R_cam * (P_world - T_cam)
        points_camera = torch.matmul(R_cam.unsqueeze(1), (points_world - T_cam.unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
        
        # [PRE-PCA NOISE INJECTION 3D]
        points_camera_noisy = points_camera.clone()
        if pre_pca_noise_params is not None:
            # 1. Depth Noise (Z-axis in camera frame)
            # Noise model: sigma = const + slope * depth
            depth = points_camera[..., 2]
            std_const = pre_pca_noise_params.get('depth_std_const', 0.0)
            std_slope = pre_pca_noise_params.get('depth_std_slope', 0.0)
            
            if std_const > 0 or std_slope > 0:
                depth_sigma = std_const + std_slope * torch.abs(depth)
                z_noise = torch.randn_like(depth) * depth_sigma
                if noise_active_mask is not None:
                     z_noise = z_noise * noise_active_mask.unsqueeze(-1)
                points_camera_noisy[..., 2] += z_noise
                
            # 2. Lateral Background Noise (X, Y in camera frame)
            lat_std = pre_pca_noise_params.get('lateral_std', 0.0)
            if lat_std > 0:
                xy_noise = torch.randn_like(points_camera[..., :2]) * lat_std
                if noise_active_mask is not None:
                     xy_noise = xy_noise * noise_active_mask.unsqueeze(-1).unsqueeze(-1)
                points_camera_noisy[..., :2] += xy_noise
                
            # 3. Outliers (3D)
            # Add random points around the object to simulate background segmentation spillover
            if 'outlier_prob' in pre_pca_noise_params and pre_pca_noise_params['outlier_prob'] > 0:
                prob = pre_pca_noise_params['outlier_prob']
                outlier_range = pre_pca_noise_params.get('outlier_range', 0.2)
                mask = torch.rand(points_camera.shape[:2], device=self.device) < prob

                # Only apply outliers where noise is active
                if noise_active_mask is not None:
                      mask = mask & noise_active_mask.unsqueeze(-1)
                
                # Perturb existing points heavily to make them outliers
                # Shift them by a random vector in [0, range]
                perturbation = torch.rand((mask.sum(), 3), device=self.device) * outlier_range
                points_camera_noisy[mask] += perturbation

        # mean_3d_cam: [num_envs, 3], eigvals_3d: [num_envs, 3], eigvecs_3d: [num_envs, 3, 3]
        mean_3d_cam, eigvals_3d, eigvecs_3d, valid_3d = compute_weighted_pca(points_camera_noisy, weights)
        
        # 7. Generate 3D Sigma Points in CAMERA FRAME with structured noise
        if noise_params is not None:
            sigma_points_3d_cam = generate_structured_noisy_sigma_points(
                mean_3d_cam, eigvals_3d, eigvecs_3d,
                pos_noise=noise_params.get('pos_3d', 0),
                scale_noise=noise_params.get('scale_3d', 0),
                rot_noise=noise_params.get('rot_3d', 0),
                alpha=alpha
            )
        else:
            sigma_points_3d_cam = generate_sigma_points(mean_3d_cam, eigvals_3d, eigvecs_3d, alpha=alpha)
        
        # Transform noisy sigma points back to World Frame (to maintain tracker output consistency)
        # P_world = R_cam^T * P_cam + T_cam
        sigma_points_3d = torch.matmul(R_cam.transpose(1, 2).unsqueeze(1), sigma_points_3d_cam.unsqueeze(-1)).squeeze(-1) + T_cam.unsqueeze(1)
        
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
        
        return sigma_points_2d, mean_2d, eigvals_2d, eigvecs_2d, weights, points_2d, sigma_points_3d, is_valid

    def transform_sigma_points_to_robot(self, sigma_points_3d, base_pos, base_quat, 
                                          cam_translation, cam_rotation, 
                                          camera_params):
        """
        Transforms sigma points from World Frame to Robot Frames (Base, Camera, Image).
        
        Args:
            sigma_points_3d (Tensor): [N, M, 3] Sigma points in World Frame.
            base_pos (Tensor): [N, 3] Robot base position.
            base_quat (Tensor): [N, 4] Robot base orientation.
            cam_translation (Tensor): [N, 3] Camera translation in Base Frame.
            cam_rotation (Tensor): [N, 3, 3] Camera rotation matrix in Base Frame.
            camera_params (dict): Camera intrinsics.
            
        Returns:
            sigma_base (Tensor): [N, M, 3] in Base Frame.
            sigma_camera (Tensor): [N, M, 3] in Camera Frame.
            sigma_image (Tensor): [N, M, 2] in Image Plane. 
        """
        num_envs = sigma_points_3d.shape[0]
        num_points = sigma_points_3d.shape[1]
        
        # Unpack camera params
        cam_fx = camera_params['fx']
        cam_fy = camera_params['fy']
        cam_cx = camera_params['cx']
        cam_cy = camera_params['cy']
        
        # 1. World -> Base
        delta = sigma_points_3d - base_pos.unsqueeze(1) # [N, M, 3]
        
        # quat_rotate_inverse expects [N, 4] and [N, 3]
        # We need to flatten our sigma points to apply the rotation
        # Expand quat to match num_points
        base_quat_expanded = base_quat.unsqueeze(1).expand(-1, num_points, -1).reshape(-1, 4)
        delta_flat = delta.reshape(-1, 3)
        
        sigma_points_base_flat = quat_rotate_inverse(base_quat_expanded, delta_flat)
        sigma_points_base = sigma_points_base_flat.view(num_envs, num_points, 3)
        
        # 2. Base -> Camera
        # Camera Frame: Optical Frame (Right-Down-Forward)
        # Position relative to camera optical center
        delta_cam = sigma_points_base - cam_translation.unsqueeze(1) # [N, M, 3]
        
        # Rotate into camera frame
        # cam_rotation is [N, 3, 3]
        # delta_cam is [N, M, 3]
        # We want res = R * delta^T  => [N, 3, 3] * [N, 3, M] = [N, 3, M]
        res = torch.bmm(cam_rotation, delta_cam.transpose(1, 2))
        sigma_points_camera = res.transpose(1, 2) # [N, M, 3]
        
        # 3. Camera -> Image
        x = sigma_points_camera[:, :, 0]
        y = sigma_points_camera[:, :, 1]
        z = sigma_points_camera[:, :, 2]
        
        # Avoid div by zero
        z_safe = torch.where(z < 1e-5, torch.ones_like(z) * 1e-5, z)
        
        # Project
        # Note: cam_fx, cam_fy, cam_cx, cam_cy are [N, 1]
        u = (x / z_safe) * cam_fx + cam_cx
        v = (y / z_safe) * cam_fy + cam_cy
        
        sigma_points_image = torch.stack([u, v], dim=-1)
        
        return sigma_points_base, sigma_points_camera, sigma_points_image

