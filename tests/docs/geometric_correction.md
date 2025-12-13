# Sim2Real 统计对齐：基于投影几何的权重修正 (Jacobian Weighting)

**摘要**：在仿真到真机 (Sim2Real) 的迁移过程中，针对点云或网格采样数据的统计特征（如 PCA 主成分）计算存在分布不匹配问题。本文档阐述了如何通过引入基于投影几何的修正权重（Jacobian Weights），将仿真中基于 3D 表面积的积分转换为等价于现实中基于 2D 图像区域的积分，从而消除“密度漂移”现象。

---

## 1. 问题陈述：积分域的不匹配 (Domain Mismatch)

在计算物体的几何中心（均值 $\mu$）和协方差矩阵（$\Sigma$）时，仿真环境与真实环境采用了不同的采样策略，导致了本质上的积分域差异：

1.  **现实环境 (Real-World)**:
    我们在图像掩码（Mask）上进行像素级采样。这在数学上等价于在 **2D 图像平面 (Image Plane)** 上进行均匀积分：
    $$\mu_{real} = \int_{Image} \mathbf{p}_{2d} \, dA_{2d}$$
    其中 $dA_{2d}$ 是图像平面的面积微元。

2.  **仿真环境 (Simulation)**:
    通常做法是在 **3D 网格表面 (Mesh Surface)** 上均匀采样点，然后投影到 2D。这在数学上等价于在 3D 表面上进行均匀积分：
    $$\mu_{sim} = \int_{Mesh} \text{Proj}(\mathbf{p}_{3d}) \, dA_{3d}$$
    其中 $dA_{3d}$ 是物体表面的面积微元。

**偏差来源**：由于透视投影的非线性特性，3D 表面上均匀分布的点在投影到 2D 平面后，其密度分布不再均匀。具体表现为：**倾斜表面（Tilted Surfaces）和深处物体（Distant Parts）在 2D 图像上的采样密度过高**。若不加修正，PCA 主轴会被错误地“拉”向这些高密度区域。

---

## 2. 几何推导 (Geometric Derivation)

为了使仿真统计量与现实对齐，我们需要找到一个权重函数 $w(\mathbf{p})$，使得 3D 表面积分近似于 2D 图像积分。这意味着我们需要寻找 2D 面积微元与 3D 表面微元之间的映射关系（即 Jacobian 行列式）：

$$dA_{2d} \approx J \cdot dA_{3d}$$

该映射关系由两个物理过程决定：**投影缩短 (Foreshortening)** 和 **透视缩放 (Perspective Scaling)**。

### 2.1 投影缩短效应 (Cosine Law)
当 3D 表面法线 $\vec{n}$ 与视线方向 $\vec{v}$ 不平行时，单位表面积在垂直于视线的平面上的有效投影面积会减小。根据朗伯余弦定律 (Lambert's Cosine Law)：

$$dA_{\perp} = dA_{3d} \cdot \cos\theta = dA_{3d} \cdot (\hat{n} \cdot \hat{v})$$

* $\hat{n}$: 表面法向量（单位向量）。
* $\hat{v}$: 从表面点指向相机光心的视线向量（单位向量）。
* $\theta$: 法线与视线的夹角。

### 2.2 透视缩放效应 (Inverse Square Law)
根据针孔相机模型，物体在图像平面上的大小与深度的平方成反比。设焦距为 $f$，点到相机的深度为 $z$，则面积缩放关系为：

$$dA_{2d} = dA_{\perp} \cdot \left( \frac{f}{z} \right)^2$$

### 2.3 综合权重公式
联立上述两步，得到 3D 微元到 2D 微元的变换关系：

$$dA_{2d} = dA_{3d} \cdot \underbrace{\left( \frac{f^2 (\hat{n} \cdot \hat{v})}{z^2} \right)}_{\text{Jacobian Weight } w}$$

由于焦距 $f$ 对于同一幅图像是常数，且后续计算会对权重进行归一化处理，因此我们只需关注比例项。最终定义的 **几何修正权重 (Geometric Correction Weight)** 为：

$$w_i = \frac{\hat{n}_i \cdot \hat{v}_i}{z_i^2}$$

---

## 3. 统计学解释 (Statistical Interpretation)

从蒙特卡洛积分 (Monte Carlo Integration) 的角度来看，这是典型的 **重要性采样 (Importance Sampling)** 应用。

* **目标分布 (Target Distribution)**: $p(x) \propto \text{Uniform}(Image)$。
* **建议分布 (Proposal Distribution)**: $q(x) \propto \text{Uniform}(Mesh)$。
* **重要性权重 (Importance Weights)**: 为了用从 $q(x)$ 采样的样本去估计 $p(x)$ 的期望，必须引入权重 $w(x) = p(x)/q(x)$。

在这里，$w_i = \frac{\hat{n}_i \cdot \hat{v}_i}{z_i^2}$ 正是该重要性权重。

通过对仿真中的每个采样点应用此权重，我们人为地降低了那些“虽然在 3D 中面积很大，但在 2D 图像中仅占据很少像素”的区域（如侧面或远端表面）的影响力。

**结论**：应用此修正后，仿真中的加权均值 $\mu_{weighted}$ 和加权协方差 $\Sigma_{weighted}$ 将是真实世界中基于 Mask 统计量的**无偏估计 (Unbiased Estimator)**。