这份系统整理将基于 Action Chunking with Temporal Aggregation (ACT) 理论，为你构建一个针对四足机器人盲区抓取的纯 RL 解决方案。
该方案的核心哲学是：利用“过去的高置信度预测”来平滑“现在的低置信度观测”。
一、 核心架构定义
1. 网络输出 (Policy Architecture)
我们将 Policy 的输出空间从“单步动作”改为“动作轨迹片段”。
Chunk Size ($k$): 设定为固定值，推荐 $k=10$ (对于 50Hz 控制频率，相当于预测未来 0.2s)。
动作维度: 假设底层控制接口为 $u \in \mathbb{R}^4$ ($v_x, v_y, \omega, \text{pitch}$)。
Policy 输出: $\mathbf{A}_t \in \mathbb{R}^{k \times 4}$。
$$\mathbf{A}_t = \{ \mathbf{a}_t^0, \mathbf{a}_t^1, \dots, \mathbf{a}_t^{k-1} \}$$
其中 $\mathbf{a}_t^i$ 表示：在时刻 $t$，预测的时刻 $t+i$ 的动作。
2. 执行逻辑 (Temporal Aggregation)
使用 统一权重 (Uniform Weighting) 进行滑动窗口集成。
在时刻 $t$，机器人实际执行的动作 $\bar{\mathbf{u}}_t$ 是过去 $k$ 次 Policy 推理结果的平均值：

$$\bar{\mathbf{u}}_t = \frac{1}{k} \sum_{j=0}^{k-1} \mathbf{a}_{t-j}^j$$
$\mathbf{a}_{t}^0$: 当前时刻 $t$ 预测的第 0 步。
$\mathbf{a}_{t-1}^1$: 上一时刻 $t-1$ 预测的第 1 步 (也是针对时刻 $t$ 的)。
$\dots$
$\mathbf{a}_{t-(k-1)}^{k-1}$: $k-1$ 时刻前预测的第 $k-1$ 步 (也是针对时刻 $t$ 的)。
二、 训练策略设计 (Training Strategy)
为了让这套架构生效，必须配套以下训练技巧。
1. Visual Dropout (盲区模拟)
为了强迫 Policy 学会“未雨绸缪”（即：准确预测 $t+5, t+9$ 的动作，而不仅是 $t+0$），我们人为引入观测丢失。
数学表达:
设原始观测为 $\mathbf{o}_t$。训练时输入网络的观测 $\tilde{\mathbf{o}}_t$ 服从伯努利分布：
$$m_t \sim \text{Bernoulli}(1 - p)$$
$$\tilde{\mathbf{o}}_t = m_t \cdot \mathbf{o}_t + (1 - m_t) \cdot \mathbf{o}_{blind}$$
$p$: Dropout 概率，推荐 0.1 (10%)。
$\mathbf{o}_{blind}$: 全 -1 向量或零向量。
作用: 当 $m_t=0$ 时，Policy 无法根据当前 $\mathbf{o}_t$ 做出正确决策，它只能寄希望于**“过去时刻预测的未来动作”**被执行。因此，它会学会在 $m_t=1$ (看得见) 的时候，输出极其精准的长程轨迹。
2. Smoothness Reward (平滑性奖励)
为了防止 Policy 输出的序列 $\mathbf{A}_t$ 内部剧烈抖动，必须约束轨迹的二阶导数或一阶差分。
数学表达:
$$r_{smooth} = - \lambda \frac{1}{k-1} \sum_{i=0}^{k-2} \| \mathbf{a}_t^{i+1} - \mathbf{a}_t^i \|^2$$
$\lambda$: 权重系数，需根据你的 Reward Scale 调整。
作用: 强迫 Policy 输出符合物理惯性的平滑曲线 (Spline)，而不是离散的跳变点。
三、 为什么能解决盲区？(数学证明)
我们来推导一下，当物体突然进入盲区时，执行动作 $\bar{\mathbf{u}}_t$ 的信噪比 (SNR) 变化。
设定场景
时刻 $T$: 物体刚好进入盲区。
时刻 $t < T$: 视野清晰，Policy 预测准确，误差为 0。
时刻 $t \ge T$: 视野全黑，Policy 输出纯噪声 $\epsilon \sim \mathcal{N}(0, \sigma^2)$。
1. 在时刻 $T$ (刚进盲区)
机器人执行的动作是：

$$\bar{\mathbf{u}}_T = \frac{1}{k} \big( \underbrace{\mathbf{a}_T^0}_{\text{当前预测}} + \underbrace{\mathbf{a}_{T-1}^1}_{\text{1步前预测}} + \dots + \underbrace{\mathbf{a}_{T-(k-1)}^{k-1}}_{\text{k-1步前预测}} \big)$$
项分析:
$\mathbf{a}_T^0$: 此时已盲，输出为噪声 $\epsilon$。
$\mathbf{a}_{T-1}^1, \dots, \mathbf{a}_{T-(k-1)}^{k-1}$: 这些都是在 $T$ 之前生成的预测，当时物体可见。因此这些项是准确的抓取动作 $\mathbf{u}^*$。
合成结果:
$$\bar{\mathbf{u}}_T = \frac{1}{k} \epsilon + \frac{k-1}{k} \mathbf{u}^*$$
结论: 噪声被稀释了 $k$ 倍，有效信号保留了 $\frac{k-1}{k}$。如果 $k=10$，90% 的动作成分依然是正确的。
2. 在时刻 $T + \delta$ (深入盲区 $\delta$ 步)
假设 $\delta < k$。

$$\bar{\mathbf{u}}_{T+\delta} = \frac{1}{k} \big( \underbrace{\sum_{j=0}^{\delta} \epsilon_j}_{\text{盲区后的预测(噪声)}} + \underbrace{\sum_{j=\delta+1}^{k-1} \mathbf{a}_{T+\delta-j}^j}_{\text{盲区前的预测(准确)}} \big)$$
有效信号占比: $\frac{k - 1 - \delta}{k}$。
物理意义: 随着时间推移，旧的“准确记忆”逐渐过期移出窗口，动作会慢慢退化为噪声。
但关键在于: 只要 $k$ 足够大（例如覆盖 0.4s），机器人完全有足够的时间在有效信号衰减完之前，完成闭合夹爪的动作。
四、 代码实现 Checklist
请按照以下步骤修改或者新增代码：
Modify Policy Output Head:
Python
# 假设 hidden_dim 是 MLP 最后一层
# action_dim = 4 (vx, vy, w, pitch)
# chunk_size = 10 # 这个数值写在Config中
self.action_head = nn.Linear(hidden_dim, action_dim * chunk_size)


Modify PPO Buffer:
存储的 actions 形状应为 [num_envs, chunk_size * action_dim]。
Reward 计算保持不变，环境 Step 保持不变（环境只拿集成后的单步动作算 Reward）。
Implement Temporal Aggregation (Inference):
Python
class ActionIntegrator:
    def __init__(self, envs, chunk_size, action_dim):
        # 维护一个 buffer: [num_envs, chunk_size, chunk_size, action_dim]
        # 这里的维度设计是为了并行处理
        self.action_history = torch.zeros(num_envs, chunk_size, chunk_size, action_dim)
        self.ptr = 0

    def add_and_aggregate(self, new_chunk_batch):
        """
        new_chunk_batch: [num_envs, chunk_size, action_dim]
        """
        # 1. 存入新预测
        # 我们只需要存针对未来 k 步的预测
        # 这一步稍微有点绕，为了性能，建议只存 new_chunk
        # 并在聚合时进行对角线索引
        pass
        # (简化的逻辑如下，非并行优化版)

# 极简非并行逻辑 (逻辑演示)
# actions_queue = deque(maxlen=k)
# actions_queue.append(new_chunk)
# current_action = mean([chunk[i] for i, chunk in enumerate(reversed(actions_queue))])


Add Visual Dropout (During Training):
Python
# 在 get_nav_commands中新增
if torch.rand(1) < 0.1:  #伪代码
    obs[:, :visual_dim] = -1.0 # 屏蔽视觉部分


Add Smoothness Penalty (In PPO Loss):
Python
# new_chunk: [Batch, k, 4] 
diff = new_chunk[:, 1:, :] - new_chunk[:, :-1, :] #伪代码
loss_smooth = torch.mean(diff ** 2)
total_reward -= 0.01 * loss_smooth


这一套组合拳下来，你就在纯 RL 的框架内，用严谨的概率统计方法解决了“物体恒常性”丢失的问题。不需要任何状态机或卡尔曼滤波。
