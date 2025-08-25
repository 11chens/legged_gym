import torch
from scipy.spatial.transform import Rotation as R_np
import numpy as np
import cv2

class camera_sensor:
    fix_extrinsics = False  # if True, the camera extrinsics are fixed, otherwise they are randomized
    fix_intrinsics = True  # if True, the camera intrinsics are fixed, otherwise they are randomized
    img_width = 1280
    img_height = 720
    class intrinsics: # Intrinsics parameters
        # Zed mini, HD720 mode
        horizontal_fov = 82 
        fx = 731.995849609375
        fy = 731.995849609375
        cx = 620.0855102539062
        cy = 362.5731201171875

    class extrinsics: # Extrinsics parameters
        translation = [0.35, 0.0, 0.0] # forward, left, upper
        angles = [0.0, 0.0, 0.0] # yaw, pitch, roll


class CameraSensor:
    """
    用于批量处理的相机模型类，支持基座坐标系、相机坐标系和图像坐标系之间的相互转换。

    坐标系说明：
    - 基座坐标系（Base）：机器人参考坐标系，X 轴指向机器人前进方向，Y 轴指向机器人左侧，Z 轴指向机器人上方。
    - 相机坐标系（Camera）：相机光学坐标系，Z 轴指向相机前方（朝向拍摄场景），X 轴指向图像宽度方向的右侧，Y 轴指向图像高度方向的下侧。
    - 图像坐标系（Image）：图像平面的 2D 坐标系，以左上角为原点，u 方向对应图像宽度，v 方向对应图像高度。

    本类通过预设或随机生成的内参和外参，提供基座→相机、相机→图像、图像→相机、相机→基座等转换函数，方便进行前向和反向几何变换。
    """
    def __init__(self, batch_size, cfg=None, device='cpu'):
        self.batch_size = batch_size
        self.cfg = cfg
        self.fix_extrinsics = cfg.fix_extrinsics
        self.fix_intrinsics = cfg.fix_intrinsics
        self.intrinsics_cfg = cfg.intrinsics
        self.extrinsics_cfg = cfg.extrinsics
        self.img_width = self.cfg.img_width
        self.img_height = self.cfg.img_height
        self.device = device

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
        初始化相机内参。这里默认使用固定的焦距和主点，

        内参包括：
        - fx, fy：焦距（单位：像素），控制 x 和 y 方向的放大比例。
        - cx, cy：主点（单位：像素），通常位于图像中心。
        """
        self.fx = self.intrinsics_cfg.fx
        self.fy = self.intrinsics_cfg.fy
        self.cx = self.intrinsics_cfg.cx
        self.cy = self.intrinsics_cfg.cy

    def _init_random_intrinsics(self):
        """
        初始化相机内参。

        内参包括：
        - fx, fy：焦距（单位：像素），控制 x 和 y 方向的放大比例。
        - cx, cy：主点（单位：像素），通常位于图像中心。
        """
        # 以下是随机初始化内参的示例，随机范围可根据实际情况调整：
        self.fx = torch.rand(self.batch_size, device=self.device) * 100 + 80   # 随机生成 [80,180] 范围的 fx
        self.fy = torch.rand(self.batch_size, device=self.device) * 100 + 80   # 随机生成 [80,180] 范围的 fy
        self.cx = torch.full((self.batch_size,), self.img_width / 2, device=self.device)  # 主点 cx 位于图像宽度中点
        self.cy = torch.full((self.batch_size,), self.img_height / 2, device=self.device) # 主点 cy 位于图像高度中点

    def _init_fixed_extrinsics(self):
        # TODO: visulize cam pos, isaacgym img, rays: FOV, HOV
        # 随机采样欧拉角（单位：度）
        yaw = torch.ones(self.batch_size, 1, device=self.device) * self.extrinsics_cfg.angles[0]
        pitch = torch.ones(self.batch_size, 1, device=self.device) * self.extrinsics_cfg.angles[1]
        roll = torch.ones(self.batch_size, 1, device=self.device) * self.extrinsics_cfg.angles[2]
        self.angles = torch.cat([yaw, pitch, roll], dim=-1)
        self.R = torch.empty((self.batch_size, 3, 3), device=self.device)
        for i in range(self.batch_size):
            # 使用 SciPy 生成欧拉角对应的旋转矩阵
            rmat = R_np.from_euler('ZYX', self.angles[i].cpu().numpy(), degrees=True).as_matrix()
            # 将基座坐标系与相机坐标系对齐的固定矩阵
            r_align = torch.tensor([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=torch.float32)
            self.R[i] = r_align @ torch.tensor(rmat, dtype=torch.float32)

        # 随机采样相机光心在基座坐标系中的位置 T = [dx, dy, dz]
        dx = torch.ones(self.batch_size, 1, device=self.device) * self.extrinsics_cfg.translation[0]
        dy = torch.ones(self.batch_size, 1, device=self.device) * self.extrinsics_cfg.translation[1]
        dz = torch.ones(self.batch_size, 1, device=self.device) * self.extrinsics_cfg.translation[2]
        self.T = torch.cat([dx, dy, dz], dim=-1)

    def _init_random_extrinsics(self):
        """
        初始化相机外参（位姿）。

        外参由旋转矩阵 R 和位移向量 T 组成：
        - 旋转矩阵 R：描述基座坐标系到相机坐标系的旋转，这里通过随机采样 yaw（偏航）、pitch（俯仰）、roll（翻滚）角生成。
          yaw ∈ [-10°, 10°]，pitch ∈ [-30°, 30°]，roll ∈ [-10°, 10°]。
          旋转矩阵按顺序采用 ZYX 欧拉角，并额外与 r_align 相乘以将基座坐标系 (X 前, Y 左, Z 上) 转换为相机坐标系 (X 右, Y 下, Z 前)。
        - 位移向量 T：描述相机光心在基座坐标系中的位置，这里随机生成：
          dx ∈ [0.3, 0.6]，dy ∈ [-0.3, 0.3]，dz ∈ [-0.3, 0.3]。
        """
        # 随机采样欧拉角（单位：度）
        yaw = torch.rand(self.batch_size, 1, device=self.device) * 20 - 10   # 偏航角范围 [-10°, 10°]
        pitch = torch.rand(self.batch_size, 1, device=self.device) * 40 - 20 # 俯仰角范围 [-20, 20]
        roll = torch.rand(self.batch_size, 1, device=self.device) * 20 - 10  # 翻滚角范围 [-10°, 10°]
        self.angles = torch.cat([yaw, pitch, roll], dim=-1)
        self.R = torch.empty((self.batch_size, 3, 3), device=self.device)
        for i in range(self.batch_size):
            # 使用 SciPy 生成欧拉角对应的旋转矩阵
            rmat = R_np.from_euler('ZYX', self.angles[i].cpu().numpy(), degrees=True).as_matrix()
            # 将基座坐标系与相机坐标系对齐的固定矩阵
            r_align = torch.tensor([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=torch.float32)
            self.R[i] = r_align @ torch.tensor(rmat, dtype=torch.float32)

        # 随机采样相机光心在基座坐标系中的位置 T = [dx, dy, dz]
        dx = torch.rand(self.batch_size, 1, device=self.device) * 0.3 + 0.3   # x 方向位移范围 [0.3, 0.6]
        dy = torch.rand(self.batch_size, 1, device=self.device) * 0.2 - 0.1   # y 方向位移范围 [-0.1, 0.1]
        dz = torch.rand(self.batch_size, 1, device=self.device) * 0.4 - 0.2   # z 方向位移范围 [-0.2, 0.2]
        self.T = torch.cat([dx, dy, dz], dim=-1)


    def base_to_camera(self, P_base: torch.Tensor) -> torch.Tensor:
        """
        将基座坐标系中的点转换到相机坐标系。

        参数：
            P_base (Tensor): 形状为 [B, 3] 的基座坐标点集合，其中 B 为批量大小。

        返回：
            Tensor: 形状为 [B, 3] 的相机坐标点集合。

        公式：
            P_cam = R * (P_base - T)
        其中 R 为旋转矩阵，T 为相机光心在基座坐标系中的位置。
        """
        # 计算点相对于相机光心的偏移量
        delta = P_base - self.T  # [B, 3]
        # 施加旋转，得到相机坐标系下的点
        P_camera = torch.bmm(self.R, delta.unsqueeze(-1)).squeeze(-1)  # [B, 3]
        return P_camera

    def camera_to_image(self, P_camera: torch.Tensor) -> torch.Tensor:
        """
        将相机坐标系中的点投影到图像平面，并得到归一化的像素坐标。

        参数：
            P_camera (Tensor): 形状为 [B, 3] 的相机坐标点集合。

        返回：
            Tensor: 形状为 [B, 2] 的归一化图像坐标，其中每个坐标范围在 [0,1]。当点在视野外或 Z<=0 时，对应位置设为 -1。
        """
        # 拆分相机坐标 X、Y、Z 分量
        X, Y, Z = P_camera[:, 0:1], P_camera[:, 1:2], P_camera[:, 2:3]
        # 根据针孔成像模型计算像素坐标
        u = self.fx * X / Z + self.cx
        v = self.fy * Y / Z + self.cy
        # 归一化到 [0, 1] 范围
        u_norm = u / self.img_width
        v_norm = v / self.img_height

        coords = torch.cat([u_norm, v_norm], dim=-1)
        # 判断是否在视野内以及深度是否为正
        visible = (Z > 0) & (u_norm >= 0) & (u_norm <= 1) & (v_norm >= 0) & (v_norm <= 1)
        visible = visible.squeeze()
        # 对于视野外或在相机后方的点，将归一化坐标设为 -1
        coords[~visible] = -1.0
        return coords

    def transform(self, P_base: torch.Tensor):
        """
        从基座坐标系到相机坐标系再到图像坐标系的完整变换。

        参数：
            P_base (Tensor): 形状为 [B, 3] 的基座坐标点集合。

        返回：
            Tuple[Tensor, Tensor]: 第一个返回值为相机坐标系中的点 [B,3]，第二个为归一化图像坐标 [B,2]。
        """
        P_camera = self.base_to_camera(P_base)
        P_image = self.camera_to_image(P_camera)
        return P_camera, P_image

    def image_to_camera(self, uv_norm: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        将归一化的图像坐标和深度值反投影到相机坐标系。

        参数：
            uv_norm (Tensor): 形状为 [B, 2] 的归一化图像坐标，范围在 [0,1]。
            depth (Tensor): 形状为 [B] 的深度值，即 Z 分量。

        返回：
            Tensor: 形状为 [B, 3] 的相机坐标点集合。
        """
        u, v = uv_norm[:, 0], uv_norm[:, 1]
        # 将归一化坐标转换为像素坐标
        u_pix = u * self.img_width
        v_pix = v * self.img_height

        # 根据反投影公式计算相机坐标
        x = (u_pix - self.cx) * depth / self.fx
        y = (v_pix - self.cy) * depth / self.fy
        z = depth
        return torch.stack([x, y, z], dim=-1)  # [B, 3]

    def camera_to_base(self, P_camera: torch.Tensor) -> torch.Tensor:
        """
        将相机坐标系中的点转换回基座坐标系。

        参数：
            P_camera (Tensor): 形状为 [B, 3] 的相机坐标点集合。

        返回：
            Tensor: 形状为 [B, 3] 的基座坐标点集合。

        公式：
            P_base = R^T * P_camera + T
        其中 R^T 为旋转矩阵的转置，T 为相机光心在基座坐标系中的位置。
        """
        # 先乘以旋转矩阵的转置进行逆旋转，再加上相机光心位置
        return torch.bmm(self.R.transpose(1, 2), P_camera.unsqueeze(-1)).squeeze(-1) + self.T  # [B, 3]

    def inverse_transform(self, uv_norm: torch.Tensor, depth: torch.Tensor):
        """
        从图像归一化坐标和深度值反推回相机坐标系和基座坐标系。

        参数：
            uv_norm (Tensor): 形状为 [B, 2] 的归一化图像坐标。
            depth (Tensor): 形状为 [B] 的深度值。

        返回：
            Tuple[Tensor, Tensor]: 第一个返回值为相机坐标系中的点 [B,3]，第二个为基座坐标点 [B,3]。
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
        在屏幕上可视化单个点在相机视图中的投影。
        如果归一化图像坐标包含 -1，则说明点在视野外或位于相机后方，此时在图像中央显示提示信息；
        否则绘制点的位置，并在图像上叠加显示基座坐标、相机坐标和归一化图像坐标。

        参数：
            P_base: 长度为 3 的数组，表示基座坐标系中的点。
            P_camera: 长度为 3 的数组，表示相机坐标系中的点。
            P_image: 长度为 2 的数组，表示归一化图像坐标。
            img_width: 图像宽度（像素）。
            img_height: 图像高度（像素）。

        返回：
            bool: 如果按下 ESC 键，则返回 False，表示退出；否则返回 True。
        """
        image = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        
        if (P_image == -1).any():
            cv2.putText(image, "Out of view or behind camera", 
                        (10, self.img_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            u_pixel = int(P_image[0] * self.img_width)
            v_pixel = int(P_image[1] * self.img_height)
            
            # 绘制投影点
            cv2.circle(image, (u_pixel, v_pixel), 5, (0, 0, 255), -1)
            
            # 叠加显示归一化图像坐标、基座坐标和相机坐标
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
    随机生成基座坐标点，对相机坐标和图像坐标进行正向和逆向变换，验证转换的正确性。

    参数：
        camera (CameraSensor): 相机对象。
        num_points (int): 测试的点的数量。

    返回：
        float: 所有测试点中的最大重建误差。
    """
    np.random.seed(42)
    max_error = 0.0

    for _ in range(num_points):
        # 随机生成基座坐标点
        dx = np.random.uniform(0.5, 6.0)
        dy = np.random.uniform(-2.0, 2.0)
        dz = np.random.uniform(-1.0, 2.0)
        P_base = np.array([[dx, dy, dz]])
        P_base_tensor = torch.from_numpy(P_base).to(dtype=torch.float32)

        # 正向变换：基座 → 相机 → 图像
        P_camera, P_image = camera.transform(P_base_tensor)
        P_camera = P_camera[0]
        P_image = P_image[0]

        # 如果投影点在视野外，则跳过
        if (P_image == -1).any():
            print(f"[跳过] 点 {P_base} 在视野外或相机后方")
            continue

        u_norm, v_norm = P_image
        depth = P_camera[2].unsqueeze(0)
        # 扩展维度以满足 inverse_transform 的输入形状
        u_norm = u_norm.unsqueeze(0).unsqueeze(0)
        v_norm = v_norm.unsqueeze(0).unsqueeze(0)
        uv_norm = torch.cat([u_norm, v_norm], dim=-1)
        # 逆向变换：图像 → 相机 → 基座
        P_camera_recon, P_base_recon = camera.inverse_transform(uv_norm, depth)

        # 计算重建误差
        error = torch.norm(P_base_tensor - P_base_recon)
        max_error = max(max_error, error)

        if error > 1e-4:
            print(f"[警告] 点 {P_base} 重建误差为: {error:.6f}")
        else:
            print(f"[正确] 点 {P_base} 变换一致")

    # 输出最大误差
    print("\n 完成")
    print(f"最大误差: {max_error:.6e}")
    return float(max_error)

if __name__ == "__main__":
    cam = CameraSensor(cfg=camera_sensor, batch_size=1)
    # dx, dy, dz = 2.0, 1.0, -0.2
    # P_base = torch.tensor([[dx, dy, dz]])
    # P_camera, P_image = cam.transform(P_base)
    # print(f"基座坐标: {P_base}")
    # print(f"相机坐标: {P_camera}")
    # print(f"归一化图像坐标: {P_image}")
    # # visualize(P_base, P_camera, P_image, cam.img_width, cam.img_height)
    verify_camera_transform(cam, num_points=20)