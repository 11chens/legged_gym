import torch

def compute_occlusion_mask(points_2d, depths, valid_mask, grid_res=64):
    """
    Compute visibility mask based on a coarse Z-buffer.
    """
    N, M, _ = points_2d.shape
    device = points_2d.device
    
    occ_mask = torch.ones((N, M), device=device)
    if not valid_mask.any():
        return occ_mask.unsqueeze(-1)

    # 1. Map normalized [0, 1] to grid coordinates
    grid_x = torch.clamp((points_2d[..., 0] * grid_res).long(), 0, grid_res - 1)
    grid_y = torch.clamp((points_2d[..., 1] * grid_res).long(), 0, grid_res - 1)
    grid_idx = grid_y * grid_res + grid_x 
    
    # 2. Build Z-buffer using ALL points
    min_depths = torch.full((N, grid_res * grid_res), 1e6, device=device)
    depths_masked = torch.where(valid_mask, depths, torch.full_like(depths, 1e6))
    min_depths = torch.scatter_reduce(min_depths, 1, grid_idx, depths_masked, reduce='min', include_self=True)
    
    # 3. Compare point depth with the min depth in its cell
    # No pooling: relying on structured sampling to fill cells.
    grid_min_depths = torch.gather(min_depths, 1, grid_idx)
    
    # [建模준수] Use a STRICT absolute depth threshold.
    # 0.02m (2cm) is a safe margin for projection discretization.
    occ_mask = (depths <= grid_min_depths + 0.02).float()
    
    return occ_mask.unsqueeze(-1)


def compute_visibility_weights(points, normals, camera_pos, use_geometric_weight=True):
    """
    Compute visibility weights based on back-face culling.
    """
    view_vec = camera_pos.unsqueeze(1) - points # [N, M, 3]
    dist_sq = (view_vec ** 2).sum(dim=-1) # [N, M]
    dist = torch.sqrt(dist_sq)
    view_dir = view_vec / (dist.unsqueeze(-1) + 1e-6)
    
    # Dot product: V . N
    dot_prod = (view_dir * normals).sum(dim=-1) # [N, M]
    
    # Strict Back-face culling
    visible_mask = (dot_prod > 0).float()
    
    if use_geometric_weight:
        # Shading: restores the variation in the Scan Map
        weights = visible_mask * dot_prod / (dist_sq + 1e-6)
    else:
        weights = visible_mask
        
    return weights.unsqueeze(-1)


def project_points(points_world, camera_params, camera_transform):
    """
    Project 3D world points to 2D image plane.
    
    Args:
        points_world (Tensor): [num_envs, num_points, 3]
        camera_params (Dict): Contains fx, fy, cx, cy, img_width, img_height (Tensors [num_envs, 1])
        camera_transform (Dict): Contains R [num_envs, 3, 3] and T [num_envs, 3] (World to Camera)
            Note: Usually T is position of camera in world, or translation vector?
            Standard: P_cam = R * (P_world - T_cam_pos) OR P_cam = R * P_world + T
            Let's assume standard Isaac Gym / OpenGL view matrix convention or what was used in legged_robot_nav.py
            In legged_robot_nav.py:
            P_camera = R_expanded @ (world_sigma_points - T_expanded).unsqueeze(-1)
            So T is camera position in world. R is rotation matrix (World to Camera? or Camera to World?)
            In legged_robot_nav.py:
            R = cam.R # [N, 3, 3]
            T = cam.T # [N, 3]
            delta = world_sigma_points - T_expanded
            P_camera = torch.matmul(R_expanded, delta.unsqueeze(-1))
            So R is likely World-to-Camera rotation (or inverse of Camera-to-World).
            
    Returns:
        points_2d (Tensor): [num_envs, num_points, 2] Normalized coordinates [-1, 1] or [0, 1]?
                            The config says "normalized [0, 1]" usually, but let's check.
                            In legged_robot_nav.py: u_norm = u / img_w.
                            Let's return pixel coords or normalized coords?
                            PCA should be done on normalized coords to be resolution independent.
                            Let's return normalized coords [0, 1].
    """
    R = camera_transform['R'] # [N, 3, 3]
    T = camera_transform['T'] # [N, 3]
    
    # Expand R and T
    # points_world: [N, M, 3]
    # R: [N, 3, 3] -> [N, 1, 3, 3]
    R_expanded = R.unsqueeze(1)
    T_expanded = T.unsqueeze(1)
    
    # P_camera = R * (P_world - T)
    delta = points_world - T_expanded # [N, M, 3]
    P_camera = torch.matmul(R_expanded, delta.unsqueeze(-1)).squeeze(-1) # [N, M, 3]
    
    X = P_camera[..., 0]
    Y = P_camera[..., 1]
    Z = P_camera[..., 2]
    
    fx = camera_params['fx']
    fy = camera_params['fy']
    cx = camera_params['cx']
    cy = camera_params['cy']
    img_w = camera_params['img_width']
    img_h = camera_params['img_height']
    
    # Avoid division by zero for points behind camera (Z <= 0)
    # We can mask them later or just let them be garbage as weights will be 0
    Z_safe = torch.where(Z <= 1e-5, torch.ones_like(Z) * 1e-5, Z)
    
    u = fx * X / Z_safe + cx
    v = fy * Y / Z_safe + cy
    
    # Normalize to [0, 1]
    u_norm = u / img_w
    v_norm = v / img_h
    
    points_2d = torch.stack([u_norm, v_norm], dim=-1) # [N, M, 2]
    
    # Valid mask: Z > 0 (in front of camera) AND inside image bounds [0, 1]
    valid_mask = (Z > 1e-4) & (u_norm >= 0.0) & (u_norm <= 1.0) & (v_norm >= 0.0) & (v_norm <= 1.0)
    
    return points_2d, valid_mask

def compute_weighted_pca(points, weights):
    """
    Compute weighted PCA of points (supports arbitrary dimension, e.g. 2D or 3D).
    
    Args:
        points (Tensor): [num_envs, num_points, dim] Points (e.g. dim=2 or dim=3).
        weights (Tensor): [num_envs, num_points, 1] Weights.
        
    Returns:
        mean (Tensor): [num_envs, dim] Weighted mean (center)
        eigvals (Tensor): [num_envs, dim] Eigenvalues (ascending: small to large)
        eigvecs (Tensor): [num_envs, dim, dim] Eigenvectors (columns)
        valid (Tensor): [num_envs] Boolean mask indicating if PCA is valid (sum_weights > 0)
    """
    # 1. Weighted Mean
    sum_weights = torch.sum(weights, dim=1) # [N, 1]
    valid = (sum_weights > 1e-6).squeeze(-1)
    
    sum_weights = torch.where(sum_weights < 1e-6, torch.ones_like(sum_weights), sum_weights) # Avoid div by zero
    
    mean = torch.sum(points * weights, dim=1) / sum_weights # [N, dim]
    
    # 2. Weighted Centered
    centered = (points - mean.unsqueeze(1)) * torch.sqrt(weights) # [N, M, dim]
    
    # 3. Weighted Covariance
    # cov = (X^T * X) / (sum_w - 1)
    # centered is [N, M, dim]
    # bmm: [N, dim, M] @ [N, M, dim] -> [N, dim, dim]
    cov = torch.bmm(centered.transpose(1, 2), centered) / (sum_weights.unsqueeze(-1) - 1 + 1e-6)
    
    # 4. Eigendecomposition
    # eigh returns eigenvalues in ascending order
    L, V = torch.linalg.eigh(cov) 
    
    return mean, L, V, valid

def generate_sigma_points(mean, eigvals, eigvecs, alpha=2.0):
    """
    Generate 2*dim + 1 sigma points from PCA results using all principal components.
    
    Args:
        mean (Tensor): [num_envs, dim]
        eigvals (Tensor): [num_envs, dim] (ascending order)
        eigvecs (Tensor): [num_envs, dim, dim] (columns are eigenvectors)
        alpha (float): Scaling factor
        
    Returns:
        sigma_points (Tensor): [num_envs, 2*dim + 1, dim]
        # Point 0: Mean
        # Point 1, 2: Mean +/- alpha * sqrt(lam_1) * v_1 (Largest)
        # Point 3, 4: Mean +/- alpha * sqrt(lam_2) * v_2 (2nd Largest)
        # ...
    """
    dim = mean.shape[1]
    
    # Clamp eigenvalues to be non-negative
    eigvals = torch.clamp(eigvals, min=1e-6)
    
    points = [mean]
    
    # Iterate from largest eigenvalue to smallest (descending importance)
    # eigvals are ascending, so iterate backwards
    for i in range(dim):
        idx = dim - 1 - i
        v = eigvecs[:, :, idx] # [N, dim]

        # [Canonicalize Sign]
        # Ensure the component with largest absolute value is positive.
        # This aligns the direction of v to avoid random flipping between Torch/Numpy or frames.
        # 1. Find the index of the largest absolute component for each vector in the batch
        max_abs_val, max_indices = torch.max(torch.abs(v), dim=1) # [N], [N]
        # 2. Gather the actual values at these indices to check their sign
        # gather expects index to have same dims, so unsqueeze
        max_vals = torch.gather(v, 1, max_indices.unsqueeze(1)).squeeze(1) # [N]
        # 3. Create flip mask: -1 where val < 0, +1 otherwise
        multipliers = torch.where(max_vals < 0, -torch.ones_like(max_vals), torch.ones_like(max_vals))
        # 4. Apply flip
        v = v * multipliers.unsqueeze(1)
        
        l = torch.sqrt(eigvals[:, idx:idx+1]) # [N, 1]
        
        p_plus = mean + alpha * l * v
        p_minus = mean - alpha * l * v
        
        points.append(p_plus)
        points.append(p_minus)
    
    sigma_points = torch.stack(points, dim=1) # [N, 2*dim+1, dim]
    
    return sigma_points

def generate_structured_noisy_sigma_points(mean, eigvals, eigvecs, pos_noise=0.0, scale_noise=0.0, rot_noise=0.0, alpha=2.0):
    """
    Generate sigma points with structured parametric noise.
    Supports anisotropic position noise and Rodrigues-based rotation perturbation.
    Specifically simulates depth camera characteristics (RealSense) where depth noise 
    propagates along the ray and couples with scale to preserve 2D projection.
    
    Args:
        mean: [B, dim] Weighted mean
        eigvals: [B, dim] Eigenvalues (ascending)
        eigvecs: [B, dim, dim] Eigenvectors (columns)
        pos_noise: Scalar or [dim] Tensor, additive noise level for mean
        scale_noise: Scalar, multiplicative noise level for sqrt(eigvals)
        rot_noise: Scalar, rotation noise level for eigenvectors (rad)
        alpha: Scaling factor for sigma points
    """
    dim = mean.shape[1]
    B = mean.shape[0]
    device = mean.device

    scale_coupling_ratio = 1.0

    # 1. Positional Noise (Additive & Ray-aligned for Sim2Real)
    if isinstance(pos_noise, (torch.Tensor, list)):
        if not isinstance(pos_noise, torch.Tensor):
            pos_noise = torch.tensor(pos_noise, device=device)
        
        if dim == 3 and pos_noise.shape[-1] >= 3:
            # Ray-aligned noise: Simulate D435i characteristics
            # Z-noise is high, but it propagates along the ray so U,V remains stable.
            dist_gt = torch.norm(mean, dim=-1, keepdim=True) + 1e-6
            
            noise_vec = torch.randn_like(mean) * pos_noise
            
            # Extract Z noise (Depth Drift: e.g. 0.5m)
            dz = noise_vec[:, 2:3]
            
            # Propagate Z noise along the ray: P_noisy = P_gt * (Z_gt + dZ) / Z_gt
            # This mathematically ensures X_noisy/Z_noisy == X_gt/Z_gt (constant U,V)
            # Center noisy position
            mean_noisy = mean * (1.0 + dz / (torch.abs(mean[:, 2:3]) + 1e-6))
            
            # [Depth-Scale Coupling Fix]
            # Calculate the ratio between noisy distance and GT distance to scale extent accordingly
            dist_noisy = torch.norm(mean_noisy, dim=-1, keepdim=True) + 1e-6
            scale_coupling_ratio = dist_noisy / dist_gt

            # Apply small residual XY jitter (shaking in 2D, e.g. 5mm) to the noisy mean
            mean = mean_noisy
            mean[:, :2] = mean[:, :2] + noise_vec[:, :2]
        else:
            mean = mean + torch.randn_like(mean) * pos_noise
    elif pos_noise > 0:
        # Standard Gaussian additive noise
        mean = mean + torch.randn_like(mean) * pos_noise

    # 2. Scale Noise (Multiplicative to sqrt(lambda))
    scales = torch.sqrt(torch.clamp(eigvals, min=1e-6))
    
    # Apply Depth-Scale Coupling: if depth is pushed far, scale up to maintain pixel size
    scales = scales * scale_coupling_ratio

    if scale_noise > 0:
        # Using normal distribution for simulated real-world feeling
        # This is the residual random jitter in mask estimation (e.g. 5%)
        scale_factor = 1.0 + torch.randn_like(scales, device=device) * scale_noise
        scales = scales * torch.clamp(scale_factor, min=0.1) # Avoid negative or near-zero scales

    # 3. Rotation Noise (Perturb eigenvectors)
    if rot_noise > 0:
        if dim == 3:
            # 3D: Apply Rodrigues perturbation to the whole rotation matrix
            # Generate random rotation axis
            rand_axis = torch.randn((B, 3), device=device)
            rand_axis = rand_axis / (torch.norm(rand_axis, dim=-1, keepdim=True) + 1e-6)
            
            # Generate random rotation angle
            rand_angle = torch.randn((B, 1), device=device) * rot_noise
            
            # Skew-symmetric matrix K
            K = torch.zeros((B, 3, 3), device=device)
            K[:, 0, 1] = -rand_axis[:, 2]
            K[:, 0, 2] =  rand_axis[:, 1]
            K[:, 1, 0] =  rand_axis[:, 2]
            K[:, 1, 2] = -rand_axis[:, 0]
            K[:, 2, 0] = -rand_axis[:, 1]
            K[:, 2, 1] =  rand_axis[:, 0]
            
            I = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)
            # Rodrigues Formula: R_perturb = I + sin(theta)K + (1-cos(theta))K^2
            sin_theta = torch.sin(rand_angle).unsqueeze(-1)
            cos_theta = torch.cos(rand_angle).unsqueeze(-1)
            R_perturb = I + sin_theta * K + (1 - cos_theta) * torch.bmm(K, K)
            
            # Apply perturbation: V_noisy = V_gt @ R_perturb
            eigvecs = torch.bmm(eigvecs, R_perturb)
            
        elif dim == 2:
            # 2D: Perturb the dominant axis angle
            v1_gt = eigvecs[:, :, 1] # Largest axis for dim=2
            theta = torch.atan2(v1_gt[:, 1], v1_gt[:, 0])
            theta_noisy = theta + torch.randn_like(theta) * rot_noise
            
            c = torch.cos(theta_noisy)
            s = torch.sin(theta_noisy)
            v1 = torch.stack([c, s], dim=-1)
            v2 = torch.stack([-s, c], dim=-1) # Perpendicular
            eigvecs = torch.stack([v2, v1], dim=2)

    # 4. Synthesize Points
    points = [mean]
    for i in range(dim):
        idx = dim - 1 - i
        v = eigvecs[:, :, idx]
        l = scales[:, idx:idx+1]
        points.append(mean + alpha * l * v)
        points.append(mean - alpha * l * v)
    
    return torch.stack(points, dim=1)
