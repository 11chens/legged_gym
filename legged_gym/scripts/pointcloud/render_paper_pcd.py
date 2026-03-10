import argparse
import torch
import open3d as o3d
import numpy as np
import os
import glob
import random

def create_publication_quality_pcd(points, colors=None, point_size=2.0):
    """
    Creates a high-quality visualization of a point cloud.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        # Default aesthetic color: Soft blue/teal
        pcd.paint_uniform_color([0.2, 0.5, 0.8])

    return pcd

def visualize_for_publication(pt_path, output_path=None):
    if not os.path.exists(pt_path):
        print(f"Error: File {pt_path} not found.")
        return

    print(f"Loading: {pt_path}")
    data = torch.load(pt_path, weights_only=True)

    if "points" not in data:
        print("Error: .pt file must contain 'points' key.")
        return

    points = data["points"].numpy()
    
    # Pre-processing points for better visualization
    # Optional: Center and Scale to unit sphere if needed for consistent shots
    center = points.mean(axis=0)
    points = points - center
    scale = np.max(np.linalg.norm(points, axis=1))
    points = points / scale

    pcd = create_publication_quality_pcd(points)

    # Visualization and Rendering
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Paper Quality Visualization", width=1200, height=1200, visible=True)
    vis.add_geometry(pcd)
    
    # Rendering options for better look
    opt = vis.get_render_option()
    opt.point_size = 5.0  # Slightly larger for better visibility in papers
    opt.background_color = np.asarray([1, 1, 1])  # Pure white background
    opt.light_on = True
    
    print("\nVisualizer controls:")
    print(" - Adjust view using mouse")
    print(" - Press 'S' to save a screenshot")
    print(" - Press 'Q' to exit")

    def save_screenshot(visualizer):
        if output_path:
            path = output_path
        else:
            obj_name = os.path.basename(os.path.dirname(pt_path))
            path = f"pointcloud_{obj_name}.png"
            
        visualizer.capture_screen_image(path)
        print(f"Image saved to: {os.path.abspath(path)}")
        return False

    key_to_callback = {
        ord("S"): save_screenshot,
    }

    o3d.visualization.draw_geometries_with_key_callbacks(
        [pcd],
        key_to_callback,
        window_name="Paper Quality Visualization",
        width=1200, height=1200,
        left=50, top=50
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate paper-quality point cloud images")
    parser.add_argument("path", type=str, nargs="?", default="obj_set/plastic_apple/point_cloud.pt", help="Path to the .pt file.")
    parser.add_argument("--output", type=str, help="Path to save the rendered image (e.g., plot.png).")
    parser.add_argument("--obj_root", type=str, default="/home/robot/project/legged_gym/obj_set", help="Root search path.")
    args = parser.parse_args()
    
    # Environment warning as requested previously
    print("Note: If visualization fails, run: export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6")
    
    pt_path = args.path
    if pt_path is None:
        search_pattern = os.path.join(args.obj_root, "**", "point_cloud.pt")
        candidates = glob.glob(search_pattern, recursive=True)
        if not candidates:
            print(f"No point_cloud.pt files found.")
            exit(1)
        pt_path = random.choice(candidates)
        print(f"Randomly selected for demo: {pt_path}")
    
    visualize_for_publication(pt_path, args.output)
