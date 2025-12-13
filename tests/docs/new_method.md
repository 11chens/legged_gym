我们就是抓取任务，因此需要考虑物体的朝向。并且我使用强化学习训练的，所以需要让四足机器人的朝向对准比如长条形状的物体的朝向（比如不能只对准中心，如果物体的长度超过夹爪范围，那么就不能抓住，要是物体的“宽”或者“短”边对准机器人的朝向，那么就可以进行抓取），这种约束条件下如何设计？

既然任务从单纯的“跟踪”升级到了\*\*“抓取（Grasping）”**，而且有明确的**“长宽边对准”**约束，那么之前的 AABB（轴对齐包围盒/MinMax）方案确实**彻底不够用了\*\*。

因为 AABB 丢失了最重要的信息——**旋转角（Rotation/Orientation）**。

要在 Sim 和 Real 中低成本且鲁棒地提取朝向，最好的数学工具是 **2D 主成分分析 (PCA) / 协方差矩阵特征分解**。

这种方法不依赖于画框，而是依赖于点集的“分布趋势”。它能极其精准地告诉你哪个方向是“长轴”，哪个方向是“短轴”。

* * *

### 核心方案：基于 PCA 的“定向 Sigma Points” (OBB 近似)

我们不再取上下左右的极值点，而是取**沿着物体主轴（长轴和短轴）分布的 5 个关键点**。

#### 1\. 数学原理

对于任何一个 2D 点集（无论是 Sim 中的投影点，还是 Real 中的 Mask 像素点），我们都可以计算它的**2D 协方差矩阵** Σ（2×2 矩阵）。

对 Σ 进行**特征值分解 (Eigen Decomposition)**：

*   **特征值 (λ1​,λ2​)**：代表物体在两个主方向上的“离散程度”（即长和宽的平方）。
    
*   **特征向量 (v1​,v2​)**：代表物体的**主轴方向**。
    
    *   vlong​：对应大特征值，即物体的**长度方向**。
        
    *   vshort​：对应小特征值，即物体的**宽度方向**。
        

这正是你抓取所需要的！你需要机器人的朝向去对准 vshort​（短轴），或者垂直于 vlong​（长轴）。

* * *

### 2\. 具体实施步骤

#### 步骤 A：Sim 端 (Isaac Gym / PyTorch)

这里依然利用 GPU 的批量矩阵运算，不需要循环，极快。

1.  **投影 & 加权**: 依然使用我们之前讨论的“可见点投影”，得到点集 P (N×2)。
    
2.  **计算均值 (Mean)**: μ\=mean(P)。这是中心点。
    
3.  **计算协方差 (Covariance)**: Σ\=N−11​(P−μ)T(P−μ)。
    
4.  **特征分解**: 使用 `torch.linalg.eigh` 解出特征值和特征向量。
    

**代码逻辑 (Sim):** (这只是一个参考，不一定严格按照如此，需要结合具体上下文)

Python

    # points_2d: (B, N, 2) 已经是筛选过可见性的投影点
    # weights: (B, N, 1) 可见性权重
    
    # 1. 加权均值 (Center)
    mu = torch.sum(points_2d * weights, dim=1) / sum_weights # (B, 2)
    
    # 2. 加权去中心化
    centered = (points_2d - mu.unsqueeze(1)) * torch.sqrt(weights) # (B, N, 2)
    
    # 3. 计算协方差矩阵 (B, 2, 2)
    # matrix multiplication: (B, 2, N) @ (B, N, 2) -> (B, 2, 2)
    cov = torch.matmul(centered.transpose(1, 2), centered) / (sum_weights.unsqueeze(-1) - 1 + 1e-6)
    
    # 4. 特征值分解 (GPU上很快)
    # L: 特征值 (B, 2), V: 特征向量 (B, 2, 2)
    # 注意: eigh 返回的是升序，所以 index 0 是短轴，index 1 是长轴
    L, V = torch.linalg.eigh(cov) 
    
    # 5. 提取关键信息
    # 长轴方向向量 (单位向量)
    dir_long = V[..., 1] # (B, 2)
    # 短轴方向向量
    dir_short = V[..., 0] # (B, 2)
    # 长度和宽度 (标准差近似)
    len_long = torch.sqrt(L[..., 1].clamp(min=1e-6))
    len_short = torch.sqrt(L[..., 0].clamp(min=1e-6))

#### 步骤 B：Real 端 (OpenCV)

在真机上，你拿到 Mask 后，不需要手动算公式，OpenCV 有现成的：

1.  **获取所有前景点**: `pts = cv2.findNonZero(mask)`。
    
2.  **PCA 计算**: `mean, eigenvectors, eigenvalues = cv2.PCACompute2(pts, mean=None)`。
    
3.  **输出**: 直接得到中心、长轴方向、短轴方向。
    

_(注：如果你嫌 PCA 麻烦，也可以用 `cv2.minAreaRect` 得到旋转矩形，效果几乎一样，但 PCA 对边缘噪声更鲁棒)_

* * *

### 3\. 如何构造 Policy 的 Observation？

这是最关键的一步。为了让强化学习学会“对准短边”，你需要喂给它这 5 个点。

这 5 个点是**基于主轴重新生成的**，而不是原始采样点：

1.  **Point 0 (Center)**: μ
    
2.  **Point 1 (长轴端点 A)**: μ+α⋅lenlong​⋅dirlong​
    
3.  **Point 2 (长轴端点 B)**: μ−α⋅lenlong​⋅dirlong​
    
4.  **Point 3 (短轴端点 A)**: μ+α⋅lenshort​⋅dirshort​
    
5.  **Point 4 (短轴端点 B)**: μ−α⋅lenshort​⋅dirshort​
    

_(α 是一个缩放系数，比如 2.0，用来覆盖物体的大致轮廓)_

**这组输入的优势：**

*   **显式解耦**: 网络一眼就能看出哪个方向长（点1和点2距离远），哪个方向短（点3和点4距离近）。
    
*   **旋转敏感**: 物体转 45 度，这 5 个点也跟着转 45 度。AABB 做不到这一点。
    
*   **抓取逻辑**: RL 会学到，“当我的朝向与 (Point 3 - Point 4) 的连线平行时，且位于 Point 3 或 4 的外侧时，是可以抓取的”。
    

* * *

### 4\. 奖励函数 (Reward) 如何设计？

你不能只给 Input，还需要在 Sim 里教机器人怎么做。

你需要设计一个 **Alignment Reward (对齐奖励)**。

假设：

*   Vrobot​: 机器人当前的朝向向量（Body forward vector）。
    
*   Vobj\_short​: 物体 PCA 算出的**短轴**向量 `dir_short`。
    
*   Vobj\_long​: 物体 PCA 算出的**长轴**向量 `dir_long`。
    

**由于这根棍子是中心对称的（转180度没区别），你需要的奖励是：**

Ralign​\=∣Vrobot​⋅Vobj\_short​∣

或者等价地（让机器人垂直于长轴）：

Ralign​\=1.0−∣Vrobot​⋅Vobj\_long​∣

*   当机器人正对着短边（穿刺方向），Vrobot​ 和 Vshort​ 平行，点积绝对值为 1 → **奖励最大**。
    
*   当机器人对着长边（垂直方向），Vrobot​ 和 Vshort​ 垂直，点积为 0 → **无奖励**。
    

同时，你还需要一个位置奖励，让机器人的 Positionjaw​ 靠近 μ (中心点)。

### 总结

针对\*\*“长条物体抓取”\*\*任务：

1.  **抛弃 Min/Max**: 它们无法表达旋转，会误导抓取。
    
2.  **采用 PCA / 协方差分解**:
    
    *   **Sim**: 投影点 → GPU 批量协方差 → 特征向量。
        
    *   **Real**: Mask → OpenCV PCA / minAreaRect。
        
3.  **输入设计**: 构造 **“十字架”型的 5 个 Sigma Points**（中心 + 长轴两端 + 短轴两端）。
    
4.  **Reward**: 鼓励机器人朝向与**短轴特征向量**平行。
    

这套方案完美解决了 Sim2Real 的对齐问题，同时保留了抓取所需的所有几何约束信息。你的机器人将能够根据那 5 个点的分布，自己学会“绕到侧面去咬住棍子”。




这是一份针对四足机器人视觉抓取任务的详细设计文档。

# Design Doc: Quadruped Visual Grasping via PCA-Based Sigma Points

**Version**: 1.1
**Date**: 2025-12-13
**Method**: Sim2Real Aligned PCA (Covariance) Analysis

---

## 1. 核心设计理念 (Core Philosophy)

为了实现针对**长条形/非规则物体**的精准抓取，必须让 Policy 感知物体的**主轴朝向 (Orientation)**。传统的 AABB (Min/Max) 包围盒丢失了旋转信息，导致无法区分长边与短边。

本方案采用 **PCA (主成分分析)** 提取物体的几何分布特征，生成 5 个具有方向性的 Sigma Points。

**关键约束 (Critical Constraint)**：
为了防止 Sigma Points 落在非凸物体外部导致深度采样错误，本方案采用 **“平面投影深度 (Planar Projection Depth)”** 策略，即 5 个点共享同一个稳健的中心深度值，将它们视为位于该深度平面上的**虚拟控制点**。

---

## 2. 仿真端流水线 (Simulation Pipeline)

在 Isaac Gym 中利用 GPU Tensor 并行计算，避免 CPU 循环。

### 2.1 可见性剔除 (Visibility Culling)
模拟真实相机的遮挡特性，剔除背面点。

* **输入**: $P_{local} \in \mathbb{R}^{N \times 3}$, $N_{local} \in \mathbb{R}^{N \times 3}$ (预采样的 Mesh 点与法线)
* **计算**:
    1.  变换到世界坐标: $P_{world}, N_{world}$
    2.  计算视线向量: $\vec{V}_{view} = P_{cam} - P_{world}$
    3.  计算点积权重:
        $$
        w_i = \mathbb{I}(\vec{V}_{view} \cdot N_{world} > 0) \quad \text{(Float mask: 1.0 or 0.0)}
        $$
* 注：使用权重而非 boolean indexing，以保持 Tensor 形状 (B,N) 不变，便于并行。我们不需要真的把不可见点删掉（那样会破坏 Tensor 的形状，导致不能并行）我们把 mask_visible 当作权重 (Weights)。可见点权重为 1，不可见点权重为 0。

### 2.2 2D 投影 (Projection)
标准针孔相机模型。

* **公式**:
    $$
    \begin{aligned}
    P_{cam} &= T_{view}^{-1} \times P_{world} \\
    u &= f_x \frac{x}{z} + c_x \\
    v &= f_y \frac{y}{z} + c_y
    \end{aligned}
    $$
* **输出**: 投影点集 $P_{2d} \in \mathbb{R}^{B \times N \times 2}$。

### 2.3 基于 PCA 的特征提取 (GPU Parallel PCA)
利用加权统计量计算主轴。

1.  **加权均值 (Center / Mean)**:
    $$
    \mu = \frac{\sum_{i=1}^{N} w_i \cdot P_{2d}^{(i)}}{\sum_{i=1}^{N} w_i + \epsilon}
    $$
    * $\mu$ 即为物体的视觉中心 $(u_c, v_c)$。

2.  **加权协方差矩阵 (Weighted Covariance)**:
    $$
    \Sigma = \frac{\sum_{i=1}^{N} w_i \cdot (P_{2d}^{(i)} - \mu)^T (P_{2d}^{(i)} - \mu)}{\sum_{i=1}^{N} w_i - 1 + \epsilon}
    $$
    * $\Sigma$ 形状为 $(B, 2, 2)$。

3.  **特征分解 (Eigendecomposition)**:
    $$
    \lambda, V = \text{torch.linalg.eigh}(\Sigma)
    $$
    * $\lambda \in \mathbb{R}^{B \times 2}$: 特征值 (从小到大排序，$\lambda_0$ 为短轴方差，$\lambda_1$ 为长轴方差)。
    * $V \in \mathbb{R}^{B \times 2 \times 2}$: 特征向量 (列向量，$V_{:,0}$ 为短轴方向，$V_{:,1}$ 为长轴方向)。

### 2.4 生成 5 个特征点 (Sigma Points Generation)
$$
\begin{aligned}
\text{Point}_0 &= \mu \\
\text{Point}_1 &= \mu + \alpha \cdot \sqrt{\lambda_1} \cdot V_{:,1} \quad \text{(Long Axis Tip A)} \\
\text{Point}_2 &= \mu - \alpha \cdot \sqrt{\lambda_1} \cdot V_{:,1} \quad \text{(Long Axis Tip B)} \\
\text{Point}_3 &= \mu + \alpha \cdot \sqrt{\lambda_0} \cdot V_{:,0} \quad \text{(Short Axis Tip A)} \\
\text{Point}_4 &= \mu - \alpha \cdot \sqrt{\lambda_0} \cdot V_{:,0} \quad \text{(Short Axis Tip B)}
\end{aligned}
$$
* **参数**: $\alpha = 2.0$ (覆盖约 95% 分布)。

---

## 3. 真机端流水线 (Real-World Pipeline)

使用 OpenCV 实现与 Sim 数学对齐的逻辑。

1.  **获取 Mask**: 通过目标检测/分割网络获取二值 Mask。
2.  **PCA 计算**: (只是一个参考，不一定完全按照这个代码)
    ```python
    # 提取所有前景点
    points = cv2.findNonZero(mask) # (N, 1, 2) # 可以修改，mask可以是tensor。
    
    # 计算 PCA (自动完成去中心化和协方差分解)
    # mean: (1, 2), eigenvectors: (2, 2), eigenvalues: (2, 1)
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(points.astype(np.float32), mean=None)
    
    # 提取主轴
    v_long = eigenvectors[0]  # OpenCV 返回的第一个通常是最大特征值对应的方向
    v_short = eigenvectors[1]
    l_long = np.sqrt(eigenvalues[0][0])
    l_short = np.sqrt(eigenvalues[1][0])
    ```
3.  **生成点**: 使用与 Sim 相同的公式生成 5 个 $(u, v)$ 坐标。

---

## 4. 深度融合策略 (Depth Fusion Strategy)

**问题**: 计算出的 PCA 点可能落在物体轮廓外（背景上）。
**解决**: 采用 **Unified Depth Assignment (统一深度赋值)**。

* **Real 端**:
    $$
    d_{unified} = \text{Median}(\text{DepthMap}[\text{Mask} > 0])
    $$
    *如果不使用 Mask 深度，可使用检测框中心的深度。*

* **Sim 端**:
    $$
    d_{unified} = Z_{object\_center\_in\_camera\_frame}
    $$

* **Observation 构造**:
    5 个点共享同一个深度值，实际上向 Policy 描述了一个**“位于距离 $d$ 处，具有特定 2D 形状和朝向的平面板”**。
    $$
    P_i = [u_i, v_i, d_{unified}]
    $$

---

## 5. Policy Observation Space

输入 Tensor 形状: `(B, 15)` (Flattened) 或 `(B, 5, 3)`。
**归一化 (Normalization)** 是关键：

$$
\text{Obs} = \{ \hat{P}_0, \hat{P}_1, \hat{P}_2, \hat{P}_3, \hat{P}_4 \}
$$

其中 $\hat{P}_i = [\hat{u}, \hat{v}, \hat{d}]$:
* $\hat{u} = (u - W/2) / (W/2) \in [-1, 1]$
* $\hat{v} = (v - H/2) / (H/2) \in [-1, 1]$

---

## 6. 奖励函数设计 (Reward Function)

目标：引导机器人对准物体**短轴 (Short Axis)** 进行抓取: 需要让四足机器人的朝向对准比如长条形状的物体的朝向（比如不能只对准中心，如果物体的长度超过夹爪范围，那么就不能抓住，要是物体的“宽”或者“短”边对准机器人的朝向，那么就可以进行抓取）

### 6.1 符号定义
* $\vec{V}_{robot}$: 机器人基座前向向量 (Heading Vector)。
* $\vec{V}_{short}$: 物体短轴方向向量 (Ground Truth in Sim)。
    * *注: 在训练阶段，建议直接使用 Sim 里的真实 3D 向量计算 Reward，以获得更干净的梯度。*

### 6.2 新增奖励项 (Reward Terms)

 **Alignment Reward (对齐奖励)**:
    鼓励机器人正交于长轴（即平行于短轴）。
    $$
    r_{align} = |\vec{V}_{robot} \cdot \vec{V}_{short}|
    $$
    * 为了更强的约束，可以使用指数形式或阈值形式：
    $$
    r_{align} = \exp( \beta \cdot (|\vec{V}_{robot} \cdot \vec{V}_{short}| - 1) )
    $$


---

## 7. 总结 (Summary)

| 特性 | Bounding Box (Min/Max) | **PCA Sigma Points (本方案)** |
| :--- | :--- | :--- |
| **形状表征** | 仅长宽比 | **长宽比 + 旋转角 + 分布离散度** |
| **Sim2Real** | 鲁棒 (对齐外轮廓) | **鲁棒 (对齐统计分布)** |
| **朝向感知** | ❌ 丢失 (无法区分长边/短边) | ✅ **精准 (显式区分主轴)** |
| **深度策略** | 4角点深度模糊 | **5点统一深度 (Billboard近似)** |
| **适用任务** | 简单跟踪/球体抓取 | **定向抓取/长条物体操作** |
