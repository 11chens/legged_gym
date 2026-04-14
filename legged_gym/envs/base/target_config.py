
class TargetCfg:
    class shape:
        type = "mixed" # "sphere", "cuboid", "cylinder", "mixed"
        # Dimensions
        radius = 0.04 # 4cm radius
        height = 0.20 # 20cm height (for cylinder)
        # For cuboid: x, y, z
        dims = [0.05, 0.20, 0.05] 
        
        # For mixed
        types = ["cuboid", "sphere", "box", "ycb", "cylinder_well"]
        # types = ["cylinder", "cuboid", "sphere", "box", "ellipsoid", "ycb"]
        # types = ["box", "ycb"]
        # Randomization Ranges
        radius_range = [0.02, 0.04] # [m]
        # height_range = [0.05, 0.07]
        height_range = [0.15, 0.2]
        # height_range = [0.05, 0.08]
        dims_range = [[0.04, 0.06], [0.15, 0.2], [0.04, 0.06]] # [[], [0.15, 0.2], []]
        
        # Ellipsoid dimensions (Short logic task)
        ellipsoid_dims_range = [[0.03, 0.10], [0.03, 0.10], [0.03, 0.10]] # [min, max] for x, y, z (diameters)
        
        # Box dimensions for Place task (Large Cuboid)
        # box_dims_range = [[0.03, 0.3], [0.03, 0.3], [0.2, 0.35]] # [min, max] for x, y, z
        box_dims_range = [[0.15, 0.2], [0.15, 0.3], [0.10, 0.20]] # [min, max] for x, y, z
        # box_dims_range = [[0.2, 0.2], [0.2, 0.2], [0.15, 0.35]] # [min, max] for x, y, z
        
        # Place Task Parameters
        place_clearance = 0.2 # [m] Height above the box edge for release
        
    class perception:
        num_sample_points = 1000
        use_geometric_weight = True
        alpha_range = [1.0, 1.5] # Sigma points scaling factor range
        debug_timer = False
        debug_info = False

        # Raw Point Noise (Simulating Sensor/Segmentation Errors BEFORE PCA)
        # This simulates that the input point cloud to PCA is not perfect surface samples
        add_pre_pca_noise = True
        pre_pca_noise = {
            'pixel_std': 0.00, # Noise strength for 2D normalized coordinates (e.g. 0.01 = 1% of image width)
            'depth_std_slope': 0.00, # Depth noise = depth * slope (e.g. 5% depth error)
            'depth_std_const': 0.00, # Constant depth noise term (m)
            'lateral_std': 0.00, # 3D Lateral noise (m), simulating partial background inclusion
            'outlier_prob': 0.0, # 5% of points are random outliers
            'outlier_range': 0.0, # Outliers are within + 20cm of the object center
        }

    class init:
        # Randomization ranges
        pos_x_range = [1.0, 3.0] # Distance from robot
        pos_y_range = [-1.0, 1.0]
        place_prob = 0.4  # Probability of Place task
        vertical_prob = 0.01  # Probability of vertical placement, we encourage horizontal placement, to help robot learn short-end grasping.
        
        # Orientation randomization
        randomize_orientation = True
        # If True, random rotation. If False, fixed.
        # We can define specific modes if needed, e.g. "horizontal_only"
