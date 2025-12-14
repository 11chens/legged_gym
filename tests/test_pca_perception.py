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

import sys
import os
import time

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

def test_pca_pipeline():
    device = 'cpu'
    num_envs = 1
    num_points = 2000 
    
    print("--- Starting PCA Perception Test (Modular Tracker) ---")
    
    # Define test cases: (Name, ShapeType, ShapeParams, Position, Orientation)
    # Orientation is (x, y, z, w)
    
    # 1. Vertical Cylinder
    # 2. Horizontal Cylinder (Rotated 90 deg around Y) -> q = (0, sin(45), 0, cos(45)) = (0, 0.707, 0, 0.707)
    # 3. Tilted Cuboid (Rotated 45 deg around Z) -> q = (0, 0, sin(22.5), cos(22.5))
    
    q_identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)
    q_horizontal = torch.tensor([[0.0, 0.7071068, 0.0, 0.7071068]], device=device)
    q_tilted_z = torch.tensor([[0.0, 0.0, 0.3826834, 0.9238795]], device=device) # 45 deg around Z
    
    test_cases = [
        {
            "name": "Cylinder_Vertical",
            "type": "cylinder",
            "params": torch.tensor([[0.2, 1.0]], device=device),
            "pos": torch.tensor([[0.0, 0.0, 0.0]], device=device),
            "quat": q_identity
        },
        {
            "name": "Cylinder_Horizontal",
            "type": "cylinder",
            "params": torch.tensor([[0.2, 1.0]], device=device),
            "pos": torch.tensor([[0.0, 0.0, 0.0]], device=device),
            "quat": q_horizontal
        },
        {
            "name": "Cuboid_Tilted_45Z",
            "type": "cuboid",
            "params": torch.tensor([[0.2, 1.0, 0.2]], device=device),
            "pos": torch.tensor([[0.0, 0.0, 0.0]], device=device),
            "quat": q_tilted_z
        }
    ]

    # Setup Camera (Same as before)
    cam_pos = torch.tensor([[3.0, 0.0, 1.0]], device=device)
    
    # Manual R (Look roughly at origin from 3,0,1)
    R = torch.tensor([[
        [0.0, -1.0, 0.0], # X_cam
        [0.0, 0.0, -1.0], # Y_cam
        [-1.0, 0.0, 0.0]  # Z_cam
    ]], device=device)
    
    T = cam_pos
    camera_transform = {'R': R, 'T': T}
    
    camera_params = {
        'fx': torch.tensor([[500.0]], device=device),
        'fy': torch.tensor([[500.0]], device=device),
        'cx': torch.tensor([[320.0]], device=device),
        'cy': torch.tensor([[240.0]], device=device),
        'img_width': 640,
        'img_height': 480
    }

    for case in test_cases:
        print(f"\n=== Testing {case['name']} ===")
        
        # Initialize Tracker
        tracker = PCATargetTracker(
            shape_type=case['type'],
            shape_params=case['params'],
            num_envs=num_envs,
            device=device,
            num_sample_points=num_points
        )
        
        # Compare Correction OFF vs ON
        modes = [("OFF", False), ("ON", True)]
        
        for mode_name, use_geo in modes:
            print(f"\n--- Mode: Geometric Weighting {mode_name} ---")
            
            # Run Tracker
            sigma_points, mean, eigvals, eigvecs, weights, points_2d = tracker.compute_features(
                object_pos=case['pos'],
                object_quat=case['quat'],
                camera_params=camera_params,
                camera_transform=camera_transform,
                use_geometric_weight=use_geo,
                debug_timer=True,
                debug_info=True
            )
            
            # Count visible points (weights > 0)
            visible_count = torch.sum(weights > 0).item()
            print(f"Visible Points: {visible_count} / {num_points} ({visible_count/num_points:.2%})")
            print(f"Mean: {mean[0].tolist()}")
            print(f"Eigenvalues: {eigvals[0].tolist()}")
            
            # Visualize
            vis_mask = weights[0, :, 0] > 0
            pts_vis = points_2d[0, vis_mask].cpu().numpy()
            sig_pts = sigma_points[0].cpu().numpy()
            
            plt.figure(figsize=(8, 6))
            plt.scatter(pts_vis[:, 0], pts_vis[:, 1], s=1, c='blue', alpha=0.5, label='Visible Points')
            
            colors = ['red', 'green', 'green', 'orange', 'orange']
            labels = ['Center', 'Long A', 'Long B', 'Short A', 'Short B']
            for i in range(5):
                plt.scatter(sig_pts[i, 0], sig_pts[i, 1], c=colors[i], s=100, marker='X', label=labels[i])
                
            center = sig_pts[0]
            long_axis = sig_pts[1] - center
            short_axis = sig_pts[3] - center
            
            plt.arrow(center[0], center[1], long_axis[0], long_axis[1], color='green', width=0.005)
            plt.arrow(center[0], center[1], short_axis[0], short_axis[1], color='orange', width=0.005)
            
            plt.xlim(0, 1)
            plt.ylim(1, 0) 
            plt.title(f"PCA Tracker: {case['name']} (Correction {mode_name})")
            plt.legend()
            plt.grid(True)
            
            output_path = f'tests/pca_tracker_{case["name"]}_{mode_name}.png'
            plt.savefig(output_path)
            print(f"Saved plot to {output_path}")
            plt.close()

if __name__ == "__main__":
    test_pca_pipeline()
