
class TargetCfg:
    class shape:
        type = "mixed" # "sphere", "cuboid", "cylinder", "mixed"
        # Dimensions
        radius = 0.04 # 4cm radius
        height = 0.20 # 20cm height (for cylinder)
        # For cuboid: x, y, z
        dims = [0.05, 0.20, 0.05] 
        
        # For mixed
        types = ["cylinder", "cuboid", "sphere"]
        # Randomization Ranges
        radius_range = [0.02, 0.05]
        height_range = [0.15, 0.25]
        dims_range = [[0.03, 0.06], [0.15, 0.25], [0.03, 0.06]] # [min, max] for x, y, z
        
        # Box dimensions for Place task (Large Cuboid)
        box_dims_range = [[0.03, 0.06], [0.03, 0.06], [0.2, 0.35]] # [min, max] for x, y, z
        
        # Place Task Parameters
        place_clearance = 0.1 # [m] Height above the box edge for release
        
    class perception:
        num_sample_points = 500
        use_geometric_weight = True
        debug_timer = False
        debug_info = False

    class init:
        # Randomization ranges
        pos_x_range = [1.0, 3.0] # Distance from robot
        pos_y_range = [-1.0, 1.0]
        place_prob = 0.3  # Probability of Place task
        vertical_prob = 0.02  # Probability of vertical placement, we encourage horizontal placement, to help robot learn short-end grasping.
        
        # Orientation randomization
        randomize_orientation = True
        # If True, random rotation. If False, fixed.
        # We can define specific modes if needed, e.g. "horizontal_only"
