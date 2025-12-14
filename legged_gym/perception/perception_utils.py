import torch

def compute_visibility_weights(points, normals, camera_pos, use_geometric_weight=True):
    """
    Compute visibility weights based on back-face culling and geometric projection.
    
    Args:
        points (Tensor): [num_envs, num_points, 3] Points in World Frame.
        normals (Tensor): [num_envs, num_points, 3] Normals in World Frame.
        camera_pos (Tensor): [num_envs, 3] Camera position in World Frame.
        use_geometric_weight (bool): If True, apply Jacobian weighting (n.v / dist^2) to simulate 2D area integral.
        
    Returns:
        weights (Tensor): [num_envs, num_points, 1] Weights.
    """
    # Vector from point to camera
    # camera_pos: [N, 3] -> [N, 1, 3]
    view_vec = camera_pos.unsqueeze(1) - points # [N, M, 3]
    
    # Distance squared
    dist_sq = (view_vec ** 2).sum(dim=-1) # [N, M]
    dist = torch.sqrt(dist_sq)
    
    # Normalize view vector
    view_dir = view_vec / (dist.unsqueeze(-1) + 1e-6)
    
    # Dot product: V . N
    # If > 0, the face is pointing towards the camera (visible)
    dot_prod = (view_dir * normals).sum(dim=-1) # [N, M]
    
    # Base visibility (Back-face culling)
    visible_mask = (dot_prod > 0).float()
    
    if use_geometric_weight:
        # Geometric Weight: (n . v) / dist^2
        # This compensates for the density difference between 3D surface sampling and 2D projection area.
        # w_i proportional to Projected Area of the surface element.
        geo_weight = visible_mask * dot_prod / (dist_sq + 1e-6)
        weights = geo_weight.unsqueeze(-1)
    else:
        weights = visible_mask.unsqueeze(-1)
    
    return weights

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
    
    # Valid mask: Z > 0 (in front of camera)
    valid_mask = Z > 1e-4
    
    return points_2d, valid_mask

def compute_weighted_pca(points_2d, weights):
    """
    Compute weighted PCA of 2D points.
    
    Args:
        points_2d (Tensor): [num_envs, num_points, 2]
        weights (Tensor): [num_envs, num_points, 1]
        
    Returns:
        mean (Tensor): [num_envs, 2] Weighted mean (center)
        eigvals (Tensor): [num_envs, 2] Eigenvalues (ascending: short, long)
        eigvecs (Tensor): [num_envs, 2, 2] Eigenvectors (columns)
        valid (Tensor): [num_envs] Boolean mask indicating if PCA is valid (sum_weights > 0)
    """
    # 1. Weighted Mean
    sum_weights = torch.sum(weights, dim=1) # [N, 1]
    valid = (sum_weights > 1e-6).squeeze(-1)
    
    sum_weights = torch.where(sum_weights < 1e-6, torch.ones_like(sum_weights), sum_weights) # Avoid div by zero
    
    mean = torch.sum(points_2d * weights, dim=1) / sum_weights # [N, 2]
    
    # 2. Weighted Centered
    centered = (points_2d - mean.unsqueeze(1)) * torch.sqrt(weights) # [N, M, 2]
    
    # 3. Weighted Covariance
    # cov = (X^T * X) / (sum_w - 1)
    # centered is [N, M, 2]
    # bmm: [N, 2, M] @ [N, M, 2] -> [N, 2, 2]
    cov = torch.bmm(centered.transpose(1, 2), centered) / (sum_weights.unsqueeze(-1) - 1 + 1e-6)
    
    # 4. Eigendecomposition
    # eigh returns eigenvalues in ascending order
    L, V = torch.linalg.eigh(cov) 
    
    return mean, L, V, valid

def generate_sigma_points(mean, eigvals, eigvecs, alpha=2.0):
    """
    Generate 5 sigma points from PCA results (using top 2 principal components).
    
    Args:
        mean (Tensor): [num_envs, dim]
        eigvals (Tensor): [num_envs, dim] (ascending order)
        eigvecs (Tensor): [num_envs, dim, dim] (columns are eigenvectors)
        alpha (float): Scaling factor
        
    Returns:
        sigma_points (Tensor): [num_envs, 5, dim]
        # Point 0: Mean
        # Point 1: Mean + alpha * sqrt(lam_long) * v_long
        # Point 2: Mean - alpha * sqrt(lam_long) * v_long
        # Point 3: Mean + alpha * sqrt(lam_short) * v_short
        # Point 4: Mean - alpha * sqrt(lam_short) * v_short
    """
    # Clamp eigenvalues to be non-negative
    eigvals = torch.clamp(eigvals, min=1e-6)
    
    # Identify principal axes (last one is largest)
    # Long axis (Largest eigenvalue)
    v_long = eigvecs[:, :, -1] # [N, dim]
    l_long = torch.sqrt(eigvals[:, -1:]) # [N, 1]
    
    # Short axis (Second largest eigenvalue)
    v_short = eigvecs[:, :, -2] # [N, dim]
    l_short = torch.sqrt(eigvals[:, -2:-1]) # [N, 1]
    
    p0 = mean
    p1 = mean + alpha * l_long * v_long
    p2 = mean - alpha * l_long * v_long
    p3 = mean + alpha * l_short * v_short
    p4 = mean - alpha * l_short * v_short
    
    sigma_points = torch.stack([p0, p1, p2, p3, p4], dim=1) # [N, 5, dim]
    
    return sigma_points
