import os
import argparse
import glob
import open3d as o3d
import numpy as np
import torch
from tqdm import tqdm

def process_obj(obj_path, save_path, num_points=2048):
    """
    Load an OBJ file, perform Poisson Disk Sampling, and save points/normals to a .pt file.
    """
    print(f"Processing: {obj_path}")
    
    # Load mesh
    # o3d.io.read_triangle_mesh handles .obj loaded with materials if mtl is present
    mesh = o3d.io.read_triangle_mesh(obj_path, enable_post_processing=True)
    
    if not mesh.has_triangles():
        # Try finding related mtl or just force load (sometimes generic loader is better)
        print(f"Warning: {obj_path} loaded with no triangles. trying separate load...")
        return

    # Ensure we have a valid mesh for sampling
    mesh.compute_vertex_normals()
    
    # Poisson Disk Sampling
    # It samples points uniformly on the surface
    # init_factor: Ratio of initial points sampled uniformly before relaxation (default 5)
    pcd = mesh.sample_points_poisson_disk(number_of_points=num_points, init_factor=5, use_triangle_normal=True)
    
    # Calculate Axis-Aligned Bounding Box (AABB) from the mesh
    # This gives the size of the object along the X, Y, Z axes
    aabb = mesh.get_axis_aligned_bounding_box()
    extent = aabb.get_extent()  # [width, height, depth] depending on axis alignment
    center = aabb.get_center()

    # Extract data
    points = np.asarray(pcd.points) - center # Center the points
    normals = np.asarray(pcd.normals)
    
    # User's requested format
    data = {
        "points": torch.from_numpy(points).float(),   # [M, 3]
        "normals": torch.from_numpy(normals).float(),  # [M, 3]
        "extent": torch.from_numpy(extent).float(),    # [3] (x, y, z lengths)
        "center": torch.from_numpy(center).float()     # [3] (x, y, z center)
    }
    
    torch.save(data, save_path)
    print(f"Saved: {save_path} | Points: {points.shape[0]} | Extent: {extent}")

def main():
    parser = argparse.ArgumentParser(description="Generate point clouds from OBJ files using Poisson Disk Sampling.")
    # Default path assumption based on workspace
    parser.add_argument("--obj_root", type=str, default="/home/robot/project/legged_gym/obj_set", help="Root directory containing OBJ sets")
    parser.add_argument("--num_points", type=int, default=1500, help="Number of points to sample")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .pt files")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.obj_root):
        print(f"Error: Directory {args.obj_root} does not exist.")
        return

    # Find all .obj files recursively
    # Pattern: obj_set/*/textured.obj usually. 
    # Use glob with recursive flag
    search_pattern = os.path.join(args.obj_root, "**", "textured.obj")
    obj_files = glob.glob(search_pattern, recursive=True)
    
    # Filter out temp files or non-mesh objs if necessary (unlikely)
    print(f"Found {len(obj_files)} .obj files in {args.obj_root}")
    
    for obj_file in tqdm(obj_files):
        folder_path = os.path.dirname(obj_file)
        # We save point_cloud.pt in the same folder
        save_path = os.path.join(folder_path, "point_cloud.pt")
        
        if os.path.exists(save_path) and not args.overwrite:
            # print(f"Skipping {save_path} (already exists)")
            continue
            
        try:
            process_obj(obj_file, save_path, args.num_points)
        except Exception as e:
            print(f"Error processing {obj_file}: {e}")

if __name__ == "__main__":
    main()
