#!/usr/bin/env python3
"""
GRU和Latent概念的详细解释和演示
"""

import torch
import torch.nn as nn
import numpy as np

def explain_gru_concepts():
    print("=" * 60)
    print("GRU (Gated Recurrent Unit) 和 Latent 概念详解")
    print("=" * 60)
    
    # 1. 设置基本参数
    batch_size = 2      # 批次大小 (比如2个机器人)
    seq_length = 5      # 序列长度 (比如5个历史时间步)
    input_size = 3      # 每个时间步的观测维度 (比如x,y,z位置)
    hidden_size = 4     # GRU隐藏状态维度
    
    print(f"批次大小: {batch_size} (假设有{batch_size}个机器人)")
    print(f"序列长度: {seq_length} (每个机器人有{seq_length}个历史观测)")
    print(f"输入维度: {input_size} (每个时间步有{input_size}个特征)")
    print(f"隐藏维度: {hidden_size} (GRU内部状态维度)")
    print()
    
    # 2. 创建示例数据
    # 形状: [batch_size, seq_length, input_size]
    obs_sequence = torch.randn(batch_size, seq_length, input_size)
    
    print("输入数据 (观测序列):")
    print(f"形状: {obs_sequence.shape}")
    print("含义: [机器人编号, 时间步, 观测特征]")
    print("具体数据:")
    for i in range(batch_size):
        print(f"  机器人 {i}:")
        for t in range(seq_length):
            print(f"    时间步 {t}: {obs_sequence[i, t, :].numpy()}")
    print()
    
    # 3. 创建GRU
    gru = nn.GRU(input_size=input_size, 
                  hidden_size=hidden_size, 
                  batch_first=True)
    
    # 4. GRU前向传播
    print("GRU处理过程:")
    print("-" * 40)
    
    # GRU返回两个值: output和hidden_state
    output, hidden_state = gru(obs_sequence)
    
    print("GRU输出详解:")
    print(f"output形状: {output.shape}")
    print(f"hidden_state形状: {hidden_state.shape}")
    print()
    
    print("output的含义:")
    print("- output包含了每个时间步的隐藏状态")
    print("- 形状为 [batch_size, seq_length, hidden_size]")
    print("- output[i, t, :] 表示第i个样本在第t个时间步的隐藏状态")
    print()
    
    # 5. 详细展示每个时间步的输出
    print("每个时间步的隐藏状态 (latent representations):")
    for i in range(batch_size):
        print(f"机器人 {i}:")
        for t in range(seq_length):
            print(f"  时间步 {t}: {output[i, t, :].detach().numpy()}")
    print()
    
    # 6. 重点解释为什么取最后一个时间步
    print("为什么取 output[:, -1, :] (最后一个时间步)?")
    print("-" * 50)
    
    last_hidden = output[:, -1, :]  # 取最后一个时间步
    print(f"最后时间步的隐藏状态形状: {last_hidden.shape}")
    print("最后时间步的隐藏状态值:")
    for i in range(batch_size):
        print(f"  机器人 {i}: {last_hidden[i, :].detach().numpy()}")
    print()
    
    print("原因解释:")
    print("1. 信息积累: GRU是递归网络,每个时间步都会:")
    print("   - 接收当前输入")
    print("   - 结合之前的记忆")
    print("   - 产生新的隐藏状态")
    print()
    print("2. 最后时间步包含最完整信息:")
    print("   - 时间步0: 只有第1个观测的信息")
    print("   - 时间步1: 包含第1-2个观测的信息")
    print("   - 时间步2: 包含第1-3个观测的信息")
    print("   - ...")
    print(f"   - 时间步{seq_length-1}: 包含第1-{seq_length}个观测的完整信息")
    print()
    
    # 7. 对比不同选择策略
    print("不同选择策略的对比:")
    print("-" * 30)
    
    # 策略1: 只取最后一个 (当前做法)
    strategy1 = output[:, -1, :]
    print(f"策略1 - 只取最后: 形状 {strategy1.shape}")
    
    # 策略2: 取所有时间步 (需要展平)
    strategy2 = output.reshape(batch_size, -1)
    print(f"策略2 - 取所有时间步: 形状 {strategy2.shape}")
    
    # 策略3: 平均池化
    strategy3 = torch.mean(output, dim=1)
    print(f"策略3 - 平均池化: 形状 {strategy3.shape}")
    
    print()
    print("各策略优缺点:")
    print("策略1 (取最后): ✓ 信息最完整 ✓ 维度固定 ✓ 计算效率高")
    print("策略2 (取所有): ✗ 维度太大 ✗ 包含冗余信息 ✗ 计算量大")
    print("策略3 (平均): ✗ 丢失时序信息 ✗ 可能模糊重要特征")
    print()
    
    # 8. 在机器人控制中的实际意义
    print("在机器人控制中的实际意义:")
    print("-" * 35)
    print("假设观测序列是机器人过去5步的状态:")
    print("- 时间步0: 机器人刚开始移动")
    print("- 时间步1: 机器人加速中")
    print("- 时间步2: 机器人转弯")
    print("- 时间步3: 机器人稳定运动")
    print("- 时间步4: 机器人当前状态")
    print()
    print("最后时间步的latent包含了:")
    print("- 运动历史模式的记忆")
    print("- 当前动态趋势")
    print("- 对未来动作的预测基础")
    print("这些信息对决策下一步动作至关重要!")

def demonstrate_shape_changes():
    print("\n" + "=" * 60)
    print("形状变化过程演示")
    print("=" * 60)
    
    # 模拟RnnActor中的实际过程
    batch_size = 3
    history_len = 10
    num_props = 48  # 机器人状态维度
    encoder_output_dim = 32
    hidden_dim = 128
    
    print("模拟 RnnActor 的实际处理过程:")
    print(f"- 批次大小: {batch_size}")
    print(f"- 历史长度: {history_len}")
    print(f"- 状态维度: {num_props}")
    print(f"- 编码输出维度: {encoder_output_dim}")
    print(f"- GRU隐藏维度: {hidden_dim}")
    print()
    
    # 1. 输入数据
    obs_hist = torch.randn(batch_size, history_len, num_props)
    print(f"1. 输入 obs_hist 形状: {obs_hist.shape}")
    print("   [批次, 历史长度, 状态维度]")
    print()
    
    # 2. 编码器处理
    encoder = nn.Sequential(nn.Linear(num_props, 128), nn.ELU(), 
                           nn.Linear(128, encoder_output_dim))
    
    # 将输入重塑为 [batch*seq, features] 进行编码
    obs_reshaped = obs_hist.view(-1, num_props)
    encoded = encoder(obs_reshaped)
    # 重新塑造回 [batch, seq, encoded_dim]
    encoded = encoded.view(batch_size, history_len, encoder_output_dim)
    
    print(f"2. 编码后形状: {encoded.shape}")
    print("   [批次, 历史长度, 编码维度]")
    print()
    
    # 3. GRU处理
    gru = nn.GRU(input_size=encoder_output_dim, 
                  hidden_size=hidden_dim, 
                  batch_first=True)
    
    latents, _ = gru(encoded)
    print(f"3. GRU输出 latents 形状: {latents.shape}")
    print("   [批次, 历史长度, 隐藏维度]")
    print()
    
    # 4. 取最后一个时间步
    last_latent = latents[:, -1, :]
    print(f"4. 最后时间步 latents[:, -1, :] 形状: {last_latent.shape}")
    print("   [批次, 隐藏维度]")
    print()
    
    # 5. 当前观测
    current_obs = obs_hist[:, -1, :]
    print(f"5. 当前观测 obs_hist[:, -1, :] 形状: {current_obs.shape}")
    print("   [批次, 状态维度]")
    print()
    
    # 6. 拼接
    actor_input = torch.cat([last_latent, current_obs], dim=-1)
    print(f"6. 拼接后 actor_input 形状: {actor_input.shape}")
    print(f"   [批次, 隐藏维度({hidden_dim}) + 状态维度({num_props}) = {hidden_dim + num_props}]")
    print()
    
    print("总结:")
    print(f"- 历史编码: {hidden_dim} 维 (压缩的历史信息)")
    print(f"- 当前状态: {num_props} 维 (详细的当前信息)")
    print(f"- 最终输入: {hidden_dim + num_props} 维 (历史+当前的完整信息)")

if __name__ == "__main__":
    explain_gru_concepts()
    demonstrate_shape_changes()