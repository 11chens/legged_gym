import torch
from scipy.spatial.transform import Rotation as R_np
import numpy as np
import cv2
from isaacgym.torch_utils import *

class camera_sensor:
    enable_camera = False  # if True, the camera sensor is enabled, otherwise it is disabled
    fix_extrinsics = False  # if True, the camera extrinsics are fixed, otherwise they are randomized
    fix_intrinsics = False  # if True, the camera intrinsics are fixed, otherwise they are randomized
    fix_img_shape = False  # if True, the image shape is fixed, otherwise it is randomized

    class intrinsics: # Intrinsics parameters
        # Zed mini, HD720 mode
        img_width = 1280
        img_height = 720
        horizontal_fov = 82.33
        fx = 731.995849609375
        fy = 731.995849609375
        cx = 620.0855102539062
        cy = 362.5731201171875

        # Zed mini, VGA mode
        img_width = 672
        img_height = 376
        horizontal_fov = 85.0
        fx = 367.0 # fx = img_width / (2 * tan(horizontal_fov/2 * pi/180))
        fy = 367.0 # fy = fx
        cx = 336.0 # cx = img_width / 2
        cy = 188.0 # cy = img_height / 2

        horizontal_fov_range = [60.0, 100.0] # [degree]
        img_height_range = [360, 720] # [pixel]
        img_width_range = [640, 1280] # [pixel]


    class extrinsics: # Extrinsics parameters
        translation = [0.4, 0.0, 0.0] # forward, left, upper
        angles = [0.0, 15.0, 0.0] # yaw, pitch, roll

        yaw_range = [-3.0, 3.0]   # [degree]
        pitch_range = [-30.0, 30.0] # [degree]
        roll_range = [-3.0, 3.0]  # [degree]
        dx_range = [0.3, 0.6]   # [m]
        dy_range = [-0.025, 0.025]   # [m]
        dz_range = [-0.2, 0.2]   # [m]

class CameraSensor:
    """
    A class to simulate a camera sensor in a 3D environment, handling camera intrinsics and extrinsics.
    The camera can transform points from the base coordinate system to the camera coordinate system,
    and then project them onto the image plane, providing normalized pixel coordinates.
    Attributes:
        batch_size (int): Number of camera instances to simulate in parallel.
        cfg (object): Configuration object containing camera parameters.
        device (str): Device to run the computations on ('cpu' or 'cuda').
        fix_extrinsics (bool): If True, camera extrinsics are fixed; otherwise, they are randomized.
        fix_intrinsics (bool): If True, camera intrinsics are fixed; otherwise, they are randomized.
        fix_img_shape (bool): If True, image shape is fixed; otherwise, it is randomized.
        intrinsics_cfg (object): Configuration for camera intrinsics.
        extrinsics_cfg (object): Configuration for camera extrinsics.
        img_height (int or Tensor): Image height in pixels. If randomized, it's a Tensor of shape [batch_size, 1].
        img_width (int or Tensor): Image width in pixels. If randomized, it's a Tensor of shape [batch_size, 1].
        fx (Tensor): Focal length in x direction (in pixels), shape [batch_size, 1] or scalar if fixed.
        fy (Tensor): Focal length in y direction (in pixels), shape [batch_size, 1] or scalar if fixed.
        cx (Tensor): Principal point x-coordinate (in pixels), shape [batch_size, 1] or scalar if fixed.
        cy (Tensor): Principal point y-coordinate (in pixels), shape [batch_size, 1] or scalar if fixed.
        R (Tensor): Rotation matrix from base frame to camera frame, shape [batch_size, 3, 3].
        T (Tensor): Translation vector of camera optical center in base frame, shape [batch_size, 3].
    """
    def __init__(self, batch_size, cfg=None, device='cpu'):
        self.batch_size = batch_size
        self.cfg = cfg
        self.fix_extrinsics = cfg.fix_extrinsics
        self.fix_intrinsics = cfg.fix_intrinsics
        self.fix_img_shape = cfg.fix_img_shape
        self.intrinsics_cfg = cfg.intrinsics
        self.extrinsics_cfg = cfg.extrinsics
        self.device = device

        if self.fix_img_shape:
            self.img_height = cfg.intrinsics.img_height
            self.img_width = cfg.intrinsics.img_width
        else:
            self.img_height_min = cfg.intrinsics.img_height_range[0]
            self.img_height_max = cfg.intrinsics.img_height_range[1]
            self.img_height = torch.randint(self.img_height_min, self.img_height_max, (self.batch_size, 1), device=self.device)
            self.img_width_min = cfg.intrinsics.img_width_range[0]
            self.img_width_max = cfg.intrinsics.img_width_range[1]
            self.img_width = torch.randint(self.img_width_min, self.img_width_max, (self.batch_size, 1), device=self.device)

        if self.fix_intrinsics:
            self._init_fixed_intrinsics()
        else:
            self._init_random_intrinsics()

        if self.fix_extrinsics:
            self._init_fixed_extrinsics()
        else:
            self._init_random_extrinsics()


    def _init_fixed_intrinsics(self):
        """
        Initialize fixed camera intrinsics.
        The intrinsics include:
        - fx, fy: Focal lengths (in pixels), controlling the scaling in the x and y directions.
        - cx, cy: Principal points (in pixels), usually located at the center of the image.
        """
        self.fx = self.intrinsics_cfg.fx
        self.fy = self.intrinsics_cfg.fy
        self.cx = self.intrinsics_cfg.cx
        self.cy = self.intrinsics_cfg.cy

    def _init_random_intrinsics(self):
        """
        Initialize random camera intrinsics.
        The intrinsics include:
        - fx, fy: Focal lengths (in pixels), controlling the scaling in the x and y directions.
        - cx, cy: Principal points (in pixels), usually located at the center of the image.
        Here, we randomly sample the horizontal field of view (FOV) within a specified range, and compute fx, fy, cx, cy accordingly.
        The image width and height are also randomly sampled within specified ranges.
        """
        self.horizontal_fov_min = self.intrinsics_cfg.horizontal_fov_range[0]
        self.horizontal_fov_max = self.intrinsics_cfg.horizontal_fov_range[1]
        self.horizontal_fov = torch_rand_float(self.horizontal_fov_min, self.horizontal_fov_max, (self.batch_size, 1), device=self.device) # [60, 100]
        self.fx = self.img_width / (2 * torch.tan(self.horizontal_fov / 2 * np.pi / 180))
        self.fy = self.fx
        self.cx = self.img_width / 2
        self.cy = self.img_height / 2

    def _init_fixed_extrinsics(self):
        """
        Initialize camera extrinsics (pose).
        The extrinsics consist of a rotation matrix R and a translation vector T:
        - Rotation matrix R: Describes the rotation from the base frame to the camera frame. Here, yaw (yaw), pitch (pitch), and roll (roll) angles are randomly sampled to generate R.
          The rotation matrix uses ZYX Euler angles in order, and is further multiplied by r_align to convert from the base frame (X forward, Y left, Z up) to the camera frame (X right, Y down, Z forward).
        - Translation vector T: Describes the camera optical center position in the base frame, randomly generated as:
        """
        yaw = torch.ones(self.batch_size, 1, device=self.device) * self.extrinsics_cfg.angles[0]
        pitch = torch.ones(self.batch_size, 1, device=self.device) * self.extrinsics_cfg.angles[1]
        roll = torch.ones(self.batch_size, 1, device=self.device) * self.extrinsics_cfg.angles[2]
        self.angles = torch.cat([yaw, pitch, roll], dim=-1)
        self.R = torch.empty((self.batch_size, 3, 3), device=self.device)
        for i in range(self.batch_size):
            rmat = R_np.from_euler('ZYX', self.angles[i].cpu().numpy(), degrees=True).as_matrix()
            r_align = torch.tensor([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=torch.float32)
            self.R[i] = r_align @ torch.tensor(rmat, dtype=torch.float32)

        dx = torch.ones(self.batch_size, 1, device=self.device) * self.extrinsics_cfg.translation[0]
        dy = torch.ones(self.batch_size, 1, device=self.device) * self.extrinsics_cfg.translation[1]
        dz = torch.ones(self.batch_size, 1, device=self.device) * self.extrinsics_cfg.translation[2]
        self.T = torch.cat([dx, dy, dz], dim=-1)

    def _init_random_extrinsics(self):
        """
        Initialize camera extrinsics (pose).
        The extrinsics consist of a rotation matrix R and a translation vector T:
        - Rotation matrix R: Describes the rotation from the base frame to the camera frame. Here, yaw (yaw), pitch (pitch), and roll (roll) angles are randomly sampled to generate R.
          The rotation matrix uses ZYX Euler angles in order, and is further multiplied by r_align to convert from the base frame (X forward, Y left, Z up) to the camera frame (X right, Y down, Z forward).
        - Translation vector T: Describes the camera optical center position in the base frame, randomly generated as:
        """
        self.yaw_min = self.extrinsics_cfg.yaw_range[0]
        self.yaw_max = self.extrinsics_cfg.yaw_range[1]
        yaw = torch_rand_float(self.yaw_min, self.yaw_max, (self.batch_size, 1), device=self.device)
        self.pitch_min = self.extrinsics_cfg.pitch_range[0]
        self.pitch_max = self.extrinsics_cfg.pitch_range[1]
        pitch = torch_rand_float(self.pitch_min, self.pitch_max, (self.batch_size, 1), device=self.device) 
        self.roll_min = self.extrinsics_cfg.roll_range[0]
        self.roll_max = self.extrinsics_cfg.roll_range[1]
        roll = torch_rand_float(self.roll_min, self.roll_max, (self.batch_size, 1), device=self.device)  
        self.angles = torch.cat([yaw, pitch, roll], dim=-1)
        self.R = torch.empty((self.batch_size, 3, 3), device=self.device)
        for i in range(self.batch_size):
            rmat = R_np.from_euler('ZYX', self.angles[i].cpu().numpy(), degrees=True).as_matrix()
            r_align = torch.tensor([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=torch.float32)
            self.R[i] = r_align @ torch.tensor(rmat, dtype=torch.float32)

        self.dx_min = self.extrinsics_cfg.dx_range[0]
        self.dx_max = self.extrinsics_cfg.dx_range[1]
        dx = torch_rand_float(self.dx_min, self.dx_max, (self.batch_size, 1), device=self.device)
        self.dy_min = self.extrinsics_cfg.dy_range[0]
        self.dy_max = self.extrinsics_cfg.dy_range[1]
        dy = torch_rand_float(self.dy_min, self.dy_max, (self.batch_size, 1), device=self.device)
        self.dz_min = self.extrinsics_cfg.dz_range[0]
        self.dz_max = self.extrinsics_cfg.dz_range[1]
        dz = torch_rand_float(self.dz_min, self.dz_max, (self.batch_size, 1), device=self.device)
        self.T = torch.cat([dx, dy, dz], dim=-1)


    def base_to_camera(self, P_base: torch.Tensor) -> torch.Tensor:
        """
        English:
        Transform points from the base coordinate system to the camera coordinate system.

        """
        # Compute the offset of the point relative to the camera optical center
        delta = P_base - self.T  # [B, 3]
        # Apply rotation to get the point in the camera coordinate system
        P_camera = torch.bmm(self.R, delta.unsqueeze(-1)).squeeze(-1)  # [B, 3]
        return P_camera

    def camera_to_image(self, P_camera: torch.Tensor) -> torch.Tensor:
        """
        Project points from the camera coordinate system to the image plane, obtaining normalized pixel coordinates.

        Args:
            P_camera (Tensor): A collection of camera coordinate points with shape [B, 3].

        Returns:
            Tensor: A collection of normalized image coordinates with shape [B, 2], where each coordinate is in the range [0, 1]. If a point is outside the field of view or Z <= 0, the corresponding position is set to -1.
        """
        # Split camera coordinates into X, Y, Z components
        X, Y, Z = P_camera[:, 0:1], P_camera[:, 1:2], P_camera[:, 2:3]
        # X.shape: [B, 1], Y.shape: [B, 1], Z.shape: [B, 1]
        # self.fx.shape: [B, 1] or scalar
        u = self.fx * X / Z + self.cx
        v = self.fy * Y / Z + self.cy
        # Normalize to [0, 1] range
        u_norm = u / self.img_width
        v_norm = v / self.img_height
        # coords.shape: [B, 2]
        coords = torch.cat([u_norm, v_norm], dim=-1)
        # Check if the points are within the field of view and if depth is positive
        visible = (Z > 0) & (u_norm >= 0) & (u_norm <= 1) & (v_norm >= 0) & (v_norm <= 1)
        visible = visible.squeeze()
        # For points outside the field of view or behind the camera, set normalized coordinates to -1
        coords[~visible] = -1.0
        return coords

    def transform(self, P_base: torch.Tensor):
        """
        Transform points from the base coordinate system to the camera coordinate system, and then project them onto the image plane.

        Args:
            P_base (Tensor): A collection of base coordinate points with shape [B, 3].

        Returns:
            Tuple[Tensor, Tensor]: The first return value is the points in the camera coordinate system [B, 3], and the second is the normalized image coordinates [B, 2].
        """
        P_camera = self.base_to_camera(P_base)
        P_image = self.camera_to_image(P_camera)
        return P_camera, P_image

    def image_to_camera(self, uv_norm: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        Project normalized image coordinates and depth values back to the camera coordinate system.

        Args:
            uv_norm (Tensor): A collection of normalized image coordinates with shape [B, 2], in the range [0, 1].
            depth (Tensor): A collection of depth values with shape [B], representing the Z component.

        Returns:
            Tensor: A collection of camera coordinate points with shape [B, 3].
        """
        u, v = uv_norm[:, 0], uv_norm[:, 1]
        # Convert normalized coordinates to pixel coordinates
        u_pix = u * self.img_width
        v_pix = v * self.img_height

        x = (u_pix - self.cx) * depth / self.fx
        y = (v_pix - self.cy) * depth / self.fy
        z = depth
        return torch.stack([x, y, z], dim=-1)  # [B, 3]

    def camera_to_base(self, P_camera: torch.Tensor) -> torch.Tensor:
        """
        Transform points from the camera coordinate system back to the base coordinate system.
        Args:
            P_camera (Tensor): A collection of camera coordinate points with shape [B, 3].
        Returns:
            Tensor: A collection of base coordinate points with shape [B, 3].
        """
        # Apply the inverse rotation and translation to get the point in the base coordinate system
        return torch.bmm(self.R.transpose(1, 2), P_camera.unsqueeze(-1)).squeeze(-1) + self.T  # [B, 3]

    def inverse_transform(self, uv_norm: torch.Tensor, depth: torch.Tensor):
        """
        Project normalized image coordinates and depth values back to the camera coordinate system and base coordinate system.

        Args:
            uv_norm (Tensor): A collection of normalized image coordinates with shape [B, 2], in the range [0, 1].
            depth (Tensor): A collection of depth values with shape [B], representing the Z component.

        Returns:
            Tuple[Tensor, Tensor]: The first return value is the points in the camera coordinate system [B, 3], and the second is the base coordinate points [B, 3].
        """
        P_camera = self.image_to_camera(uv_norm, depth)
        P_base = self.camera_to_base(P_camera)
        return P_camera, P_base

    def visualize_img(self, img):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        cv2.imshow('Image', img)
        cv2.waitKey(1)

    def visualize_img_coords(self, P_base, P_camera, P_image):
        """
        Visualize the projection of a single point in the camera view on the screen.
        If the normalized image coordinates contain -1, it indicates that the point is out of view or behind the camera,
        and a message will be displayed in the center of the image;
        otherwise, the position of the point will be drawn, and the base coordinates, camera coordinates, and normalized image coordinates will be overlaid on the image.

        Args:
            P_base: A collection of base coordinate points with shape [B, 3].
            P_camera: A collection of camera coordinate points with shape [B, 3].
            P_image: A collection of normalized image coordinates with shape [B, 2].
            img_width: The width of the image (in pixels).
            img_height: The height of the image (in pixels).

        Returns:
            bool: If the ESC key is pressed, return False to indicate exit; otherwise, return True.
        """
        image = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        
        if (P_image == -1).any():
            cv2.putText(image, "Out of view or behind camera", 
                        (10, self.img_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            u_pixel = int(P_image[0] * self.img_width)
            v_pixel = int(P_image[1] * self.img_height)
            
            cv2.circle(image, (u_pixel, v_pixel), 5, (0, 0, 255), -1)
            
            text = f"Normal P_img: ({P_image[0]:.3f}, {P_image[1]:.3f})"
            cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            text_base = f"P_base: ({P_base[0]:.3f}, {P_base[1]:.3f}, {P_base[2]:.3f})"
            cv2.putText(image, text_base, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            text_camera = f"P_cam: ({P_camera[0]:.3f}, {P_camera[1]:.3f}, {P_camera[2]:.3f})"
            cv2.putText(image, text_camera, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Camera View", image)
        
        key = cv2.waitKey(1)
        
        if key == 27:
            cv2.destroyAllWindows()
            return False 
        
        return True

def verify_camera_transform(camera: CameraSensor, num_points: int = 10) -> float:
    """
    Verify the consistency of the camera transformation by randomly generating points in the base coordinate system,
    transforming them to the camera coordinate system and image plane, and then inversely transforming them back
    to the camera and base coordinate systems. The reconstruction error is computed to assess the accuracy of
    the transformations.
    """
    np.random.seed(42)
    max_error = 0.0

    for _ in range(num_points):
        dx = np.random.uniform(0.5, 6.0)
        dy = np.random.uniform(-2.0, 2.0)
        dz = np.random.uniform(-1.0, 2.0)
        P_base = np.array([[dx, dy, dz]])
        P_base_tensor = torch.from_numpy(P_base).to(dtype=torch.float32)

        # English: Transform the point from the base coordinate system to the camera coordinate system and image plane
        P_camera, P_image = camera.transform(P_base_tensor)
        P_camera = P_camera[0]
        P_image = P_image[0]

        # If the projected point is out of view, skip it
        if (P_image == -1).any():
            print(f"[Skip] Point {P_base} is out of view or behind the camera")
            continue

        u_norm, v_norm = P_image
        depth = P_camera[2].unsqueeze(0)
        # Expand dimensions to match the input shape of inverse_transform
        u_norm = u_norm.unsqueeze(0).unsqueeze(0)
        v_norm = v_norm.unsqueeze(0).unsqueeze(0)
        uv_norm = torch.cat([u_norm, v_norm], dim=-1)
        # Inverse transform: image → camera → base
        P_camera_recon, P_base_recon = camera.inverse_transform(uv_norm, depth)

        # Compute reconstruction error
        error = torch.norm(P_base_tensor - P_base_recon)
        max_error = max(max_error, error)

        if error > 1e-4:
            print(f"[Warning] Point {P_base} reconstruction error: {error:.6f}")
        else:
            print(f"[Correct] Point {P_base} transformation is consistent")

    # Output maximum error
    print("\nDone")
    print(f"Maximum error: {max_error:.6e}")
    return float(max_error)

if __name__ == "__main__":
    cam = CameraSensor(cfg=camera_sensor, batch_size=1)
    # dx, dy, dz = 2.0, 1.0, -0.2
    # P_base = torch.tensor([[dx, dy, dz]])
    # P_camera, P_image = cam.transform(P_base)
    # print(f"Base coordinates: {P_base}")
    # print(f"Camera coordinates: {P_camera}")
    # print(f"Normalized image coordinates: {P_image}")
    # # visualize(P_base, P_camera, P_image, cam.img_width, cam.img_height)
    verify_camera_transform(cam, num_points=20)