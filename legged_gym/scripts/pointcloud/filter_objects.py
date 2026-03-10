import os
import torch
import argparse
import glob
from tqdm import tqdm
import shutil
import numpy as np

# ==========================================
# Filter Policy Definitions
# ==========================================

class FilterPolicy:
    """Base interface for filtering policies."""
    def check(self, extent):
        """
        Check if the object properties meet the criteria.
        Args:
            extent (np.array): [3] Dimensions of the object (x, y, z)
        Returns:
            (bool, str): (Passed?, Reason for failure)
        """
        raise NotImplementedError

class SizeFilterPolicy(FilterPolicy):
    """
    Standard size-based filtering policy.
    Checks if object is within specific bounds (upper and lower) for dimensions.
    """
    def __init__(self, max_dim_upper=0.15, min_dim_upper=0.10, max_dim_lower=0.05, min_dim_lower=0.03):
        self.max_dim_upper = max_dim_upper
        self.min_dim_upper = min_dim_upper 
        self.max_dim_lower = max_dim_lower
        self.min_dim_lower = min_dim_lower

    def check(self, extent):
        max_d = np.max(extent)
        min_d = np.min(extent)
        
        failures = []
        
        # Upper Bound Checks (Too Big?)
        if self.max_dim_upper is not None and max_d >= self.max_dim_upper:
            failures.append(f"MaxDim({max_d:.3f}) >= Upper({self.max_dim_upper})")

        if self.min_dim_upper is not None and min_d >= self.min_dim_upper:
            failures.append(f"MinDim({min_d:.3f}) >= Upper({self.min_dim_upper})")
            
        # Lower Bound Checks (Too Small?)
        if self.max_dim_lower is not None and max_d < self.max_dim_lower:
             failures.append(f"MaxDim({max_d:.3f}) < Lower({self.max_dim_lower})")

        if self.min_dim_lower is not None and min_d < self.min_dim_lower:
             failures.append(f"MinDim({min_d:.3f}) < Lower({self.min_dim_lower})")

        if failures:
            return False, " | ".join(failures)
        return True, "OK"

# Can add more policies here easily...
# class VolumeFilterPolicy(FilterPolicy): ...

# ==========================================
# Processing Engine
# ==========================================

def process_filtering(obj_root, policy, move_invalid_to=None):
    """
    Main driver function. Detached from specific logic.
    """
    search_pattern = os.path.join(obj_root, "**", "point_cloud.pt")
    pt_files = glob.glob(search_pattern, recursive=True)
    
    if not pt_files:
        print(f"No point_cloud.pt files found in {obj_root}. Please run test_pointcloud.py first.")
        return

    valid_objects = []
    invalid_objects = []
    missing_data_objects = []
    
    print(f"Scanning {len(pt_files)} objects in {obj_root}...")
    
    for pt_file in tqdm(pt_files):
        try:
            data = torch.load(pt_file)
            if "extent" not in data:
                missing_data_objects.append(pt_file)
                continue
                
            extent = data["extent"].numpy() # [3]
            obj_name = os.path.basename(os.path.dirname(pt_file))
            
            # --- Decoupled Logic Call ---
            is_valid, reason = policy.check(extent)
            
            if is_valid:
                valid_objects.append((obj_name, extent))
            else:
                invalid_objects.append((obj_name, extent, pt_file, reason))
                
        except Exception as e:
            print(f"Error reading {pt_file}: {e}")

    # --- Report Results ---
    print_report(valid_objects, invalid_objects, missing_data_objects)

    # --- Handle Actions ---
    if move_invalid_to and invalid_objects:
        execute_move(invalid_objects, move_invalid_to)

def print_report(valid, invalid, missing):
    print("\n" + "="*100)
    print(f"FILTERING RESULTS")
    print("="*100)
    print(f"Total Processed: {len(valid) + len(invalid) + len(missing)}")
    print(f" Passed:         {len(valid)}")
    print(f" Rejected:       {len(invalid)}")
    print(f" Data Errors:    {len(missing)}")
    print("-" * 100)
    
    if valid:
        print(f"\n[PASSED] (Compatible Objects)")
        print(f"{'Object Name':<35} | {'Dimensions (X, Y, Z) [m]'}")
        print("-" * 60)
        for name, dims in sorted(valid):
            d_str = f"[{dims[0]:.3f}, {dims[1]:.3f}, {dims[2]:.3f}]"
            print(f"{name:<35} | {d_str}")

    if invalid:
        print(f"\n[REJECTED] (Incompatible Objects)")
        print(f"{'Object Name':<35} | {'Dimensions [m]':<25} | {'Failure Reason'}")
        print("-" * 100)
        for name, dims, _, reason in sorted(invalid):
            d_str = f"[{dims[0]:.3f}, {dims[1]:.3f}, {dims[2]:.3f}]"
            # Highlight dimensions close to limit?
            print(f"{name:<35} | {d_str:<25} | {reason}")

def execute_move(invalid_objects, destination):
    print("\n" + "-"*60)
    confirm = input(f"Move {len(invalid_objects)} rejected objects to '{destination}'? (y/n): ")
    if confirm.lower() != 'y':
        print("Operation cancelled.")
        return

    if not os.path.exists(destination):
        os.makedirs(destination)
        print(f"Created directory: {destination}")

    success_count = 0
    for name, _, pt_file, _ in invalid_objects:
        src_dir = os.path.dirname(pt_file)
        dst_dir = os.path.join(destination, name)
        
        try:
            if os.path.exists(dst_dir):
                print(f"Skip {name}: Already exists in destination.")
            else:
                shutil.move(src_dir, dst_dir)
                success_count += 1
        except Exception as e:
            print(f"Error moving {name}: {e}")
            
    print(f"Successfully moved {success_count} objects.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter objects using a Size Policy.")
    parser.add_argument("--obj_root", type=str, default="/home/robot/project/legged_gym/obj_set")
    
    # Upper Bounds (Max allowed size)
    parser.add_argument("--max_dim_upper", type=float, default=0.20, help="Upper limit for largest dim (e.g. < 15cm)")
    parser.add_argument("--min_dim_upper", type=float, default=0.10, help="Upper limit for smallest dim (e.g. < 10cm)")
    
    # Lower Bounds (Min allowed size)
    parser.add_argument("--max_dim_lower", type=float, default=0.05, help="Lower limit for largest dim (e.g. > 5cm)")
    parser.add_argument("--min_dim_lower", type=float, default=0.03, help="Lower limit for smallest dim (e.g. > 3cm)")
    
    parser.add_argument("--move_to", type=str, default=None, help="Directory to move rejected objects to")
    
    args = parser.parse_args()
    
    # Instantiate the desired policy (Decoupled)
    policy = SizeFilterPolicy(
        max_dim_upper=args.max_dim_upper, 
        min_dim_upper=args.min_dim_upper,
        max_dim_lower=args.max_dim_lower,
        min_dim_lower=args.min_dim_lower
    )
    
    process_filtering(args.obj_root, policy, args.move_to)
