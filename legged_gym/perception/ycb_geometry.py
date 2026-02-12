import torch
import os
import glob
from .surface_geometry import SurfaceShape

class YCBManager:
    """
    Manages loading and serving YCB object point clouds.
    """
    def __init__(self, obj_root, device='cpu', max_cache_points=2048):
        self.device = device
        self.max_cache_points = max_cache_points
        self.loaded_data = [] # List of dicts
        self.obj_names = []
        
        self._load_objects(obj_root)
        
        # Convert list of dicts to stacked tensors for efficient indexing
        # Assumption: All .pt files have at least `max_cache_points` points.
        # If they vary, we might need to pad, but for now we assume uniform generation.
        
        if len(self.loaded_data) > 0:
            num_objs = len(self.loaded_data)
            
            # Stack points and normals: [NumObjs, MaxPoints, 3]
            # We take the first max_cache_points to ensure shape consistency
            self.points_tensor = torch.stack([d['points'][:max_cache_points] for d in self.loaded_data]).to(device)
            self.normals_tensor = torch.stack([d['normals'][:max_cache_points] for d in self.loaded_data]).to(device)
            self.extents_tensor = torch.stack([d['extent'] for d in self.loaded_data]).to(device)
            # Centers might be needed if the mesh wasn't centered
            if 'center' in self.loaded_data[0]:
                self.centers_tensor = torch.stack([d['center'] for d in self.loaded_data]).to(device)
            else:
                self.centers_tensor = torch.zeros((num_objs, 3), device=device)
                
            print(f"[YCBManager] Loaded {num_objs} objects from {obj_root}. Device: {device}")
        else:
             print(f"[YCBManager] Warning: No objects loaded from {obj_root}!")
             self.points_tensor = torch.empty(0, max_cache_points, 3, device=device)
             self.normals_tensor = torch.empty(0, max_cache_points, 3, device=device)
             self.extents_tensor = torch.empty(0, 3, device=device)

    def _load_objects(self, obj_root):
        search_pattern = os.path.join(obj_root, "**", "point_cloud.pt")
        pt_files = glob.glob(search_pattern, recursive=True)
        pt_files.sort() # Ensure deterministic order
        
        for f in pt_files:
            try:
                data = torch.load(f, map_location='cpu')
                # Validation
                if data['points'].shape[0] < self.max_cache_points:
                    print(f"Skipping {f}: Not enough points ({data['points'].shape[0]} < {self.max_cache_points})")
                    continue
                if 'extent' not in data:
                    print(f"Skipping {f}: No extent data.")
                    continue
                
                self.loaded_data.append(data)
                self.obj_names.append(os.path.basename(os.path.dirname(f)))
            except Exception as e:
                print(f"Error loading {f}: {e}")

    def get_num_objects(self):
        return len(self.loaded_data)

    def get_info(self, indices):
        """
        Get extents and centers for given object indices.
        Args:
           indices: Tensor [N] of object IDs
        Returns:
           extents: [N, 3]
           centers: [N, 3]
        """
        return self.extents_tensor[indices], self.centers_tensor[indices]

class YCBGeometry(SurfaceShape):
    def __init__(self, manager: YCBManager):
        super().__init__(manager.device)
        self.manager = manager

    def sample_surface(self, num_points, num_envs, params):
        """
        Serve points from the loaded library.
        Args:
            num_points: points to serve (must be <= manager.max_cache_points)
            num_envs: number of environments
            params: [num_envs, 1] tensor containing object indices
        """
        indices = params.view(-1).long()
        
        # Select objects: [N, MaxPts, 3]
        selected_points = self.manager.points_tensor[indices]
        selected_normals = self.manager.normals_tensor[indices]
        
        # Subsample points if needed
        # We can just take the first num_points or random shuffle
        # Since we want randomness per episode/step? 
        # Actually `sample_surface` is usually called once per reset or when we need "new" points. 
        # But for static meshes, points are fixed. 
        # To simulate sensor noise, we might want different subsets? 
        # For now, let's just take a random subset to be robust.
        
        total_available = selected_points.shape[1]
        if num_points < total_available:
            # Random indices
            # Generating unique random indices per env is expensive in a loop.
            # Fast approx: shuffle global index or just take slice if Poisson is good.
            # Let's take a random start index and wrap around, or random permutation.
            # Simplest: randperm of indices [0, total_available] -> take first num_points
            perm = torch.randperm(total_available, device=self.device)[:num_points]
            points = selected_points[:, perm, :]
            normals = selected_normals[:, perm, :]
        else:
            points = selected_points
            normals = selected_normals
            
        return points, normals
