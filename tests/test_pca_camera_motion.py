import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import isaacgym first to avoid conflicts
try:
    import isaacgym
except ImportError:
    pass

import torch
import matplotlib.pyplot as plt
import numpy as np

from legged_gym.perception.tracker import PCATargetTracker

def look_at_rotation(camera_pos, target_pos, up_vector=None):
    """
    Generate Rotation Matrix R (World -> Camera) such that Camera looks at Target.
    Convention:
    Z_cam points from Camera to Target (Forward)
    X_cam points Right
    Y_cam points Down (for image coordinates)
    """
    # Ensure inputs are tensors
    if not isinstance(camera_pos, torch.Tensor):
        camera_pos = torch.tensor(camera_pos)
    if not isinstance(target_pos, torch.Tensor):
        target_pos = torch.tensor(target_pos)
    
    if up_vector is None:
        up_vector = torch.tensor([0.0, 0.0, 1.0], device=camera_pos.device, dtype=camera_pos.dtype)
    else:
        up_vector = up_vector.to(device=camera_pos.device, dtype=camera_pos.dtype)
        
    # Forward vector (Z axis)
    forward = target_pos - camera_pos
    forward = forward / torch.norm(forward)
    
    # Right vector (X axis)
    # Handle case where forward is parallel to up
    right = torch.cross(forward, up_vector, dim=-1)
    if torch.norm(right) < 1e-6:
        # If looking straight up/down, choose arbitrary right
        right = torch.tensor([1.0, 0.0, 0.0], device=camera_pos.device, dtype=camera_pos.dtype)
    else:
        right = right / torch.norm(right)
        
    # Down vector (Y axis)
    down = torch.cross(forward, right, dim=-1)
    
    # Construct Rotation Matrix
    # Rows are X, Y, Z axes in World Frame
    R = torch.stack([right, down, forward], dim=0)
    
    return R

def test_camera_motion():
    device = 'cpu'
    num_envs = 1
    num_points = 1000
    
    print("--- Starting PCA Camera Motion Test (Horizontal Cylinder) ---")
    
    # 1. Setup Object (Static Cylinder at Origin)
    # Use Horizontal Cylinder so it looks different from different angles
    tracker = PCATargetTracker(
        shape_type='cylinder',
        shape_params=torch.tensor([[0.2, 1.0]], device=device, dtype=torch.float32), # r=0.2, h=1.0
        num_envs=num_envs,
        device=device,
        num_sample_points=num_points
    )
    
    obj_pos = torch.tensor([[0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
    # Rotate 90 deg around Y axis: (0, 0.707, 0, 0.707)
    obj_quat = torch.tensor([[0.0, 0.7071068, 0.0, 0.7071068]], device=device, dtype=torch.float32)
    
    # 2. Define Camera Trajectory (Orbit around Z axis)
    # Radius 3.0, Height 1.0
    angles = np.linspace(0, 2*np.pi, 8)[:-1] # 0, 45, 90, ...
    
    camera_params = {
        'fx': torch.tensor([[500.0]], device=device, dtype=torch.float32),
        'fy': torch.tensor([[500.0]], device=device, dtype=torch.float32),
        'cx': torch.tensor([[320.0]], device=device, dtype=torch.float32),
        'cy': torch.tensor([[240.0]], device=device, dtype=torch.float32),
        'img_width': 640,
        'img_height': 480
    }
    
    prev_mean = None
    
    for i, angle in enumerate(angles):
        print(f"\n--- Step {i}: Angle {np.degrees(angle):.1f} deg ---")
        
        # Calculate Camera Position
        x = 3.0 * np.cos(angle)
        y = 3.0 * np.sin(angle)
        z = 1.0
        cam_pos = torch.tensor([[x, y, z]], device=device, dtype=torch.float32)
        
        # Calculate Rotation
        R = look_at_rotation(cam_pos[0], obj_pos[0])
        
        camera_transform = {
            'R': R.unsqueeze(0), # [1, 3, 3]
            'T': cam_pos         # [1, 3]
        }
        
        # Run Tracker
        sigma_points, mean, eigvals, eigvecs, weights, points_2d, _ = tracker.compute_features(
            object_pos=obj_pos,
            object_quat=obj_quat,
            camera_params=camera_params,
            camera_transform=camera_transform,
            use_geometric_weight=True,
            debug_timer=False,
            debug_info=True,
        )
        
        # Check Results
        vis_count = (weights > 0).sum().item()
        mean_val = mean[0].tolist()
        
        print(f"  Cam Pos: [{x:.2f}, {y:.2f}, {z:.2f}]")
        print(f"  Visible Points: {vis_count}")
        print(f"  Projected Mean: [{mean_val[0]:.4f}, {mean_val[1]:.4f}]")
        
        # Verify change
        if prev_mean is not None:
            diff = torch.norm(mean[0] - prev_mean).item()
            print(f"  Diff from prev mean: {diff:.6f}")
            if diff < 1e-5:
                print("  WARNING: Mean did not change significantly!")
        
        prev_mean = mean[0].clone()
        
        # Visualize
        vis_mask = weights[0, :, 0] > 0
        pts_vis = points_2d[0, vis_mask].cpu().numpy()
        
        plt.figure(figsize=(6, 4))
        plt.scatter(pts_vis[:, 0], pts_vis[:, 1], s=1, c='blue', alpha=0.5)
        plt.xlim(0, 1)
        plt.ylim(1, 0) # Image coords
        plt.title(f"Angle {np.degrees(angle):.0f} deg")
        plt.grid(True)
        plt.savefig(f"tests/camera_motion_{i}.png")
        plt.close()

if __name__ == "__main__":
    test_camera_motion()
