import argparse
import open3d as o3d
import os
import glob
import random
import numpy as np

def render_mesh_for_publication(mesh_path, output_path=None):
    if not os.path.exists(mesh_path):
        print(f"Error: Mesh file {mesh_path} not found.")
        return

    print(f"Loading mesh: {mesh_path}")
    # Load mesh - open3d handles .obj with textures if .mtl and images are in the same folder
    mesh = o3d.io.read_triangle_model(mesh_path)
    # Note: read_triangle_model returns a Model object, but we might just want the mesh
    # Let's try loading as a simple mesh first for better control over visualization
    mesh_geometry = o3d.io.read_triangle_mesh(mesh_path, enable_post_processing=True)
    
    if not mesh_geometry.has_triangles():
        print("Error: Could not load mesh triangles.")
        return

    mesh_geometry.compute_vertex_normals()

    # Pre-processing: Center and scale
    center = mesh_geometry.get_center()
    mesh_geometry.translate(-center)
    scale = np.max(np.linalg.norm(np.asarray(mesh_geometry.vertices), axis=1))
    if scale > 0:
        mesh_geometry.scale(1.0 / scale, center=[0, 0, 0])

    print("\nVisualizer controls:")
    print(" - Adjust view using mouse")
    print(" - Press 'S' to save a screenshot")
    print(" - Press 'Q' to exit")

    def save_screenshot(visualizer):
        if output_path:
            path = output_path
        else:
            obj_name = os.path.basename(os.path.dirname(mesh_path))
            path = f"mesh_{obj_name}.png"
            
        visualizer.capture_screen_image(path)
        print(f"Image saved to: {os.path.abspath(path)}")
        return False

    key_to_callback = {
        ord("S"): save_screenshot,
    }

    # Setup visualization with specific options
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="YCB Mesh Visualization", width=1200, height=1200)
    vis.add_geometry(mesh_geometry)
    
    # Paper-style setup
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1]) # White background
    opt.mesh_show_back_face = True
    
    # Register callback
    vis.register_key_callback(ord("S"), save_screenshot)
    
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render YCB object mesh for publication")
    parser.add_argument("path", type=str, nargs="?", default="obj_set/plastic_peach/textured.obj", help="Path to the .obj file.")
    parser.add_argument("--output", type=str, help="Path to save the rendered image.")
    args = parser.parse_args()
    
    print("Note: If visualization fails, run: export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6")

    mesh_path = args.path
    if mesh_path is None:
        # Try to find a textured.obj in obj_set
        obj_root = "/home/robot/project/legged_gym/obj_set"
        candidates = glob.glob(os.path.join(obj_root, "**", "textured.obj"), recursive=True)
        if not candidates:
            print(f"No textured.obj files found in {obj_root}")
            exit(1)
        mesh_path = random.choice(candidates)
        print(f"Randomly selected for demo: {mesh_path}")
    
    render_mesh_for_publication(mesh_path, args.output)
