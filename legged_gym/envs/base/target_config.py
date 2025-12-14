
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
        
    class perception:
        num_sample_points = 1000
        frame = "base" # "base", "camera", "image"
        use_geometric_weight = True
        debug_timer = False
        debug_info = False
        
    class init:
        # Randomization ranges
        pos_x_range = [1.0, 3.0] # Distance from robot
        pos_y_range = [-1.0, 1.0]
        
        # Orientation randomization
        randomize_orientation = True
        # If True, random rotation. If False, fixed.
        # We can define specific modes if needed, e.g. "horizontal_only"
