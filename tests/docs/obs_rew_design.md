
Observation and Reward 设计的核心原则：保证机器人去抓取一个横着放的圆柱形水瓶，让机器人知道不能从横着的边抓而是从瓶底/瓶口抓。

### 1. Observation 设计：不要给像素，给“Robot Body Frame 下的 3D 坐标”

为了让对齐奖励跟网络输入挂钩紧密，最直接的方法是：让输入数据本身就包含明确的方向矢量信息。

建议在输入网络前，先进行坐标变换（这是确定性的数学计算，不需要网络学）：Pixel $(u, v, d)$ → Camera Frame $(x, y, z)$ → Robot Body Frame $(x_b, y_b, z_b)$

**推荐的 Policy Input (Tensor Shape: [B, 15])**

我们将这 5 个 Sigma Points 转换到机器人的**基座坐标系 (Base Frame)** 下。

- $P_{center}$ (中心): 物体中心在机器人坐标系的位置。
- $P_{head}$ (瓶口 - 长轴端点A): 基于 PCA 长轴正方向生成的点。
- $P_{tail}$ (瓶底 - 长轴端点B): 基于 PCA 长轴负方向生成的点。
- $P_{side1}$ (侧面1): 短轴端点。
- $P_{side2}$ (侧面2): 短轴端点。

### Observation Vector (15 dims):

$$
\text{Obs} = [p_{center}, p_{head}, p_{tail}, p_{side1}, p_{side2}]
$$

(其中每个 $p$ 都是 $(x, y, z)$ 坐标)

---

### 为什么这样设计极其高效？

想象机器人看到了一个横着放的水瓶：

- $P_{center}$ 在机器人正前方 1 米 $(1.0, 0, 0)$。
- $P_{head}$ 在左前方 $(1.0, 0.2, 0)$。
- $P_{tail}$ 在右前方 $(1.0, -0.2, 0)$。

网络一眼就能看到：$P_{head}$ 和 $P_{tail}$ 的 $y$ 坐标差异很大 —— 这意味着物体是横着的！如果机器人想对齐，它必须旋转，直到 $P_{head}$ 和 $P_{tail}$ 的 $y$ 坐标都接近 0（即物体纵向对着机器人）。


具体设计：

我们将这 5 个 Sigma Points 转换到机器人的**基座坐标系 (Base Frame)** 下。

- $P_{center}$ (中心): 物体中心在机器人坐标系的位置。
- $P_{head}$ (瓶口 - 长轴端点A): 基于 PCA 长轴正方向生成的点。
- $P_{tail}$ (瓶底 - 长轴端点B): 基于 PCA 长轴负方向生成的点。
- $P_{side1}$ (侧面1): 短轴端点。
- $P_{side2}$ (侧面2): 短轴端点。

### Observation Vector (15 dims):

$$
\text{Obs} = [p_{center}, p_{head}, p_{tail}, p_{side1}, p_{side2}]
$$

(其中每个 $p$ 都是 $(x, y, z)$ 坐标)

---

### 为什么这样设计极其高效？

想象机器人看到了一个横着放的水瓶：

- $P_{center}$ 在机器人正前方 1 米 $(1.0, 0, 0)$。
- $P_{head}$ 在左前方 $(1.0, 0.2, 0)$。
- $P_{tail}$ 在右前方 $(1.0, -0.2, 0)$。

网络一眼就能看到：$P_{head}$ 和 $P_{tail}$ 的 $y$ 坐标差异很大 —— 这意味着物体是横着的！如果机器人想对齐，它必须旋转，直到 $P_{head}$ 和 $P_{tail}$ 的 $y$ 坐标都接近 0（即物体纵向对着机器人）。

---

### 2. 核心逻辑：如何定义“从瓶口/瓶底抓”？

既然要抓瓶口或瓶底，那么机器人的导航目标 (Target) 就不是物体中心 $P_{center}$，而是 $P_{head}$ 或 $P_{tail}$。

你需要让 Policy 动态选择离它最近的那个端点作为目标。

**定义动态目标 $P_{target}$:**

$$
P_{target} =
\begin{cases}
P_{head}, & \text{if } \|P_{robot} - P_{head}\| < \|P_{robot} - P_{tail}\| \\
P_{tail}, & \text{otherwise}
\end{cases}
$$

---

### 3. 奖励函数设计：紧密耦合 (Tightly Coupled)

为了让机器人“意识到”必须对准长轴，我们需要设计由三个分量组成的复合奖励。

#### 3.1. 接近奖励 (Approach Reward) - 引导去端点

不要奖励靠近中心，要奖励靠近选定的端点。

$$
R_{approach} = \frac{1}{1 + \|P_{jaw} - P_{target}\|^2}
$$

- **直观理解：** 这会驱使机器人走向瓶口或瓶底，而不是走向瓶身中间。

#### 3.2. 对齐奖励 (Alignment Reward) - 引导旋转

这是最关键的。机器人机身朝向 $V_{robot}$ 必须与物体的长轴 $V_{long} = P_{head} - P_{tail}$ 平行（而不是垂直）。

$$
\text{CosSim} = \frac{V_{robot} \cdot (P_{head} - P_{tail})}{\|P_{head} - P_{tail}\|}
$$

我们希望这个值的绝对值接近 1。

$$
R_{align} = |\text{CosSim}|^k \quad (k = 2 \text{ or } 4)
$$

- **指数 $k$ 的作用：** 当 $k$ 较大时，只有非常精准的对齐才能拿到高分，稍微歪一点分数掉很快。这会强迫机器人进行精细调整。
- **直观理解：** 输入 Obs 里的 $P_{head}$ 和 $P_{tail}$ 坐标差直接决定了这个 Reward。Policy 会迅速学到：“只要把 $P_{head}$ 和 $P_{tail}$ 摆在我的正前方一条线上（X轴上），我就能拿高分。”

#### 3.3. 抓取约束奖励 (Grasp Constraint Reward)

为了防止机器人“倒着走”或者“侧身横移”过去，我们需要约束速度矢量与朝向一致。

$$
R_{velocity} = V_{linear\_velocity} \cdot V_{robot\_heading}
$$

- **结合 $R_{align}$，这确保机器人是正对着瓶口走过去的。**
