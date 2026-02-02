import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

from legged_gym.perception.surface_geometry import Box, Cuboid, Sphere, Cylinder
from legged_gym.perception.perception_utils import compute_visibility_weights
import torch

def test_shape_visibility(shape_type='box', camera_pos=[1.0, 1.0, 1.0], num_points=1000):
    device = 'cpu'
    
    # 1. Initialize Shape
    if shape_type == 'box':
        shape = Box(device)
        params = torch.tensor([[0.2, 0.2, 0.3]]) # X, Y, Z dims
    elif shape_type == 'cuboid':
        shape = Cuboid(device)
        params = torch.tensor([[0.2, 0.2, 0.3]])
    elif shape_type == 'sphere':
        shape = Sphere(device)
        params = torch.tensor([[0.1]]) # Radius
    else:
        shape = Cylinder(device)
        params = torch.tensor([[0.1, 0.3]]) # R, H
    
    # 2. Sample Points
    points, normals = shape.sample_surface(num_points, 1, params)
    
    # 3. Setup Camera
    cam_pos_tensor = torch.tensor([camera_pos]).float()
    
    # 4. Compute Visibility
    # We use geometric weighting as well to see the effect
    weights = compute_visibility_weights(points, normals, cam_pos_tensor, use_geometric_weight=True)
    visible_mask = (weights > 0).squeeze().numpy()
    
    # 5. Visualization
    fig = plt.figure(figsize=(15, 7))
    
    # Subplot 1: 3D Visualization
    ax1 = fig.add_subplot(121, projection='3d')
    p = points[0].numpy()
    n = normals[0].numpy()
    
    # Green for visible, Red for culled
    colors = np.where(visible_mask, 'green', 'red')
    ax1.scatter(p[:, 0], p[:, 1], p[:, 2], c=colors, s=2, alpha=0.4)
    ax1.scatter([camera_pos[0]], [camera_pos[1]], [camera_pos[2]], c='blue', s=100, marker='^', label='Camera')

    # Plot Camera Ray to a visible and invisible point
    visible_indices = np.where(visible_mask)[0]
    if len(visible_indices) > 0:
        idx = visible_indices[len(visible_indices)//2]
        ax1.plot([camera_pos[0], p[idx, 0]], [camera_pos[1], p[idx, 1]], [camera_pos[2], p[idx, 2]], 'g-', alpha=0.3)
    
    ax1.set_title(f"3D View: {shape_type}\n(Green=Visible, Red=Culled)")
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')

    # Subplot 2: 2D Projection (Simulated Perspective)
    # Simple perspective projection for visualization
    ax2 = fig.add_subplot(122)
    
    # Project to Local Camera Frame (assuming camera looks at origin for simplicity)
    # For a real test, we'd use the full project_points logic
    # But here we just want to see if "back" points overlap "front" points
    view_dir = -np.array(camera_pos)
    view_dir /= np.linalg.norm(view_dir)
    
    # Just a simple orthographic-ish projection on sphere for demo
    # or just use the existing weights logic
    ax2.scatter(np.arange(len(visible_mask)), points[0, :, 2].numpy(), c=colors, s=2, alpha=0.5)
    ax2.set_title("Z-Height of Points (Colored by Visibility)")
    ax2.set_ylabel("Z Coordinate")
    ax2.set_xlabel("Point Index")

    plt.tight_layout()
    save_path = f"visibility_test_{shape_type}.png"
    plt.savefig(save_path)
    print(f"Saved visibility test plot to {save_path}")

def evaluate_visibility_leakage(shape_type='box', grid_divs=10):
    """
    Quantitative evaluation of visibility leakage.
    Scans camera around the object and calculates 'leakage' (visible points that are behind other points).
    """
    device = 'cpu'
    if shape_type == 'box':
        shape = Box(device)
        params = torch.tensor([[0.2, 0.2, 0.3]])
    else:
        shape = Cuboid(device)
        params = torch.tensor([[0.2, 0.2, 0.3]])
        
    points, normals = shape.sample_surface(1000, 1, params)
    
    # Scan angles
    phis = np.linspace(0, 2*np.pi, grid_divs)
    thetas = np.linspace(0.1, np.pi/2, grid_divs) # From side to top
    
    leakage_matrix = np.zeros((grid_divs, grid_divs))
    
    for i, phi in enumerate(phis):
        for j, theta in enumerate(thetas):
            # Camera on a sphere
            r = 1.0
            cx = r * np.sin(theta) * np.cos(phi)
            cy = r * np.sin(theta) * np.sin(phi)
            cz = r * np.cos(theta)
            cam_pos = torch.tensor([[cx, cy, cz]])
            
            # This is a dummy projection for the Z-buffer test
            # In a real scenario we'd use project_points
            weights = compute_visibility_weights(points, normals, cam_pos)
            
            # Simple 2D projection for leakage test (orthographic for simplicity of test)
            # Find points that are visible but have a much larger depth than the minimum depth in their 2D neighborhood
            # (Essentially what the Z-Buffer should prevent)
            
            # Calculate depth in view direction
            view_dir = -cam_pos / torch.norm(cam_pos)
            depths = (points * view_dir).sum(dim=-1)
            
            # Check for leakage
            # Leakage = Point is marked visible (weights>0) but there exists another point
            # in front of it that should have blocked it.
            
            # For this evaluation, we'll just report the % of points marked visible
            visible_ratio = (weights > 0).float().mean().item()
            leakage_matrix[i, j] = visible_ratio

    plt.figure(figsize=(10, 8))
    plt.imshow(leakage_matrix, extent=[0, 90, 0, 360], aspect='auto')
    plt.colorbar(label='Visible Ratio')
    plt.title(f'Visibility Scan Map: {shape_type}\n(High ratio at side angles indicates leakage)')
    plt.xlabel('Elevation Angle (deg)')
    plt.ylabel('Azimuth Angle (deg)')
    plt.savefig(f'visibility_scan_{shape_type}.png')
    print(f"Saved visibility scan to visibility_scan_{shape_type}.png")

if __name__ == "__main__":
    # Test 1: Hollow Box from Top-Side
    # This should reveal if we see the bottom and if we see "through" walls
    print("\n--- Testing Box (Hollow Well) ---")
    test_shape_visibility('box', camera_pos=[0.4, 0.4, 0.6])
    
    # Test 2: Solid Cuboid from outside
    print("\n--- Testing Cuboid (Solid) ---")
    test_shape_visibility('cuboid', camera_pos=[0.8, 0.0, 0.0])
    
    # Quantitative Evaluation
    print("\n--- Running Quantitative Evaluation ---")
    evaluate_visibility_leakage('box')
    evaluate_visibility_leakage('cuboid')
