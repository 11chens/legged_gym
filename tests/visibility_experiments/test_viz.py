import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from shapes import Sphere, Cuboid, Cylinder
from visibility import check_visibility

def plot_visibility(points, is_visible, camera_pos, title):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Visible points
    visible_points = points[0, is_visible[0]].cpu().numpy()
    if len(visible_points) > 0:
        ax.scatter(visible_points[:, 0], visible_points[:, 1], visible_points[:, 2], c='g', marker='o', label='Visible')
    
    # Invisible points
    invisible_points = points[0, ~is_visible[0]].cpu().numpy()
    if len(invisible_points) > 0:
        ax.scatter(invisible_points[:, 0], invisible_points[:, 1], invisible_points[:, 2], c='r', marker='x', label='Invisible')
    
    # Camera
    cam = camera_pos[0].cpu().numpy()
    ax.scatter(cam[0], cam[1], cam[2], c='b', marker='^', s=100, label='Camera')
    
    # Draw lines from camera to a few visible points to visualize view
    if len(visible_points) > 0:
        for i in range(min(5, len(visible_points))):
            ax.plot([cam[0], visible_points[i, 0]], [cam[1], visible_points[i, 1]], [cam[2], visible_points[i, 2]], 'g--', alpha=0.3)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    # Set equal aspect ratio hack
    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([points[0, :, 0].max()-points[0, :, 0].min(), 
                          points[0, :, 1].max()-points[0, :, 1].min(), 
                          points[0, :, 2].max()-points[0, :, 2].min()]).max() / 2.0

    mid_x = (points[0, :, 0].max()+points[0, :, 0].min()) * 0.5
    mid_y = (points[0, :, 1].max()+points[0, :, 1].min()) * 0.5
    mid_z = (points[0, :, 2].max()+points[0, :, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.savefig(f'tests/visibility_experiments/{title.replace(" ", "_")}.png')
    print(f"Saved plot to tests/visibility_experiments/{title.replace(' ', '_')}.png")
    plt.close()

def test_sphere():
    print("Testing Sphere...")
    device = 'cpu'
    sphere = Sphere(device=device)
    
    num_envs = 1
    num_points = 500
    radius = 1.0
    params = torch.tensor([[radius]], device=device)
    
    points, normals = sphere.sample_surface(num_points, num_envs, params)
    
    # Camera at (3, 0, 0) looking at origin
    camera_pos = torch.tensor([[3.0, 0.0, 0.0]], device=device)
    
    is_visible = check_visibility(points, normals, camera_pos)
    
    plot_visibility(points, is_visible, camera_pos, "Sphere Visibility")

def test_cuboid():
    print("Testing Cuboid...")
    device = 'cpu'
    cuboid = Cuboid(device=device)
    
    num_envs = 1
    num_points = 500
    params = torch.tensor([[1.0, 2.0, 1.0]], device=device) # x=1, y=2, z=1
    
    points, normals = cuboid.sample_surface(num_points, num_envs, params)
    
    # Camera at (2, 2, 2)
    camera_pos = torch.tensor([[2.0, 2.0, 2.0]], device=device)
    
    is_visible = check_visibility(points, normals, camera_pos)
    
    plot_visibility(points, is_visible, camera_pos, "Cuboid Visibility")

def test_cylinder():
    print("Testing Cylinder...")
    device = 'cpu'
    cylinder = Cylinder(device=device)
    
    num_envs = 1
    num_points = 500
    params = torch.tensor([[0.5, 2.0]], device=device) # r=0.5, h=2.0
    
    points, normals = cylinder.sample_surface(num_points, num_envs, params)
    
    # Camera at (0, 3, 0) - looking at side
    camera_pos = torch.tensor([[0.0, 3.0, 0.0]], device=device)
    
    is_visible = check_visibility(points, normals, camera_pos)
    
    plot_visibility(points, is_visible, camera_pos, "Cylinder Visibility")

if __name__ == "__main__":
    test_sphere()
    test_cuboid()
    test_cylinder()
