import torch

def check_visibility(points, normals, camera_pos):
    """
    Check visibility of points using back-face culling.
    
    Args:
        points (Tensor): [num_envs, num_points, 3] Points in World Frame.
        normals (Tensor): [num_envs, num_points, 3] Normals in World Frame.
        camera_pos (Tensor): [num_envs, 3] Camera position in World Frame.
        
    Returns:
        is_visible (Tensor): [num_envs, num_points] Boolean mask. True if visible.
    """
    # Vector from point to camera
    # camera_pos: [N, 3] -> [N, 1, 3]
    view_vec = camera_pos.unsqueeze(1) - points # [N, M, 3]
    
    # Normalize view vector (optional for sign check, but good practice)
    # We only need the dot product sign, so normalization isn't strictly necessary 
    # if we just check > 0, but let's do it for correctness if we wanted angle.
    # For pure back-face culling, dot product with unnormalized vector works too 
    # as long as we check sign.
    
    # Dot product
    dot_prod = (view_vec * normals).sum(dim=-1) # [N, M]
    
    # Visible if normal points towards camera (angle < 90 deg -> dot > 0)
    is_visible = dot_prod > 0
    
    return is_visible
