import argparse
import torch
import open3d as o3d
import numpy as np
import os
import glob
import random

def visualize_pt(pt_path):
    if not os.path.exists(pt_path):
        print(f"Error: File {pt_path} not found.")
        return

    print(f"Loading: {pt_path}")
    try:
        data = torch.load(pt_path)
    except Exception as e:
        print(f"Error loading torch file: {e}")
        return

    if "points" not in data or "normals" not in data:
        print("Error: .pt file must contain 'points' and 'normals' keys.")
        print(f"Keys found: {data.keys()}")
        return

    points = data["points"].numpy()
    normals = data["normals"].numpy()
    
    print(f"Generic Stats:")
    print(f"  Points Shape: {points.shape}")
    print(f"  Normals Shape: {normals.shape}")
    
    if "extent" in data:
        print(f"  Extent (Size): {data['extent'].numpy()}")
    if "center" in data:
        print(f"  Center: {data['center'].numpy()}")

    # Create Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    
    # Paint it a uniform color (e.g., cyan/blueish)
    pcd.paint_uniform_color([0.0, 0.7, 1.0])

    print("Opening visualization window independently...")
    print("Commands:")
    print("  [Mouse] Rotate/Pan/Zoom")
    print("  [N]     Toggle normals display")
    print("  [+]     Increase point size")
    print("  [-]     Decrease point size")
    print("  [Q]     Quit")
    
    # Create coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    
    o3d.visualization.draw_geometries([pcd, coord_frame], 
                                    window_name="Point Cloud Visualization",
                                    width=1024, height=768,
                                    left=50, top=50,
                                    point_show_normal=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize PointCloud from .pt file")
    parser.add_argument("path", type=str, nargs="?", help="Path to the .pt file. If omitted, picks randomly from obj_set.")
    parser.add_argument("--obj_root", type=str, default="/home/robot/project/legged_gym/obj_set", help="Root search path if no file specified.")
    args = parser.parse_args()
    print("Before running, Please set env var:\nexport LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6")
    
    pt_path = args.path
    if pt_path is None:
        search_pattern = os.path.join(args.obj_root, "**", "point_cloud.pt")
        candidates = glob.glob(search_pattern, recursive=True)
        if not candidates:
            print(f"No point_cloud.pt files found in {args.obj_root}. Run test_pointcloud.py first?")
            exit(1)
        pt_path = random.choice(candidates)
        print(f"Randomly selected: {pt_path}")
    
    visualize_pt(pt_path)
