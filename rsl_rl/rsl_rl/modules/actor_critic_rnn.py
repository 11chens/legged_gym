# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from .common_modules import RnnStateHistoryEncoder, get_activation, mlp_factory, StateHistoryEncoder

class RnnActor(nn.Module):
    def __init__(self,
                 num_props,
                 encoder_dims,
                 actor_dims,
                 encoder_output_dim,
                 hidden_dim,
                 num_actions,
                 activation,) -> None:
        super(RnnActor, self).__init__()
        self.num_props = num_props
        self.hidden_dim = hidden_dim
        self.encoder_output_dim = encoder_output_dim
        
        # 编码器 - 将观测编码到较小的维度
        self.encoder = nn.Sequential(*mlp_factory(activation=activation,
                                   input_dims=num_props,
                                   hidden_dims=encoder_dims,
                                   out_dims=encoder_output_dim))
        
        # GRU用于记忆管理
        self.rnn = nn.GRU(input_size=encoder_output_dim,
                         hidden_size=hidden_dim,
                         batch_first=True)
        
        # Actor网络
        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                                input_dims=hidden_dim + num_props,
                                                out_dims=num_actions,
                                                hidden_dims=actor_dims))
        
        # 隐藏状态管理
        self.hidden_states = None

    def forward(self, obs_hist, masks=None, hidden_states=None):
        # 处理输入观测序列
        batch_size, seq_len, _ = obs_hist.shape
        
        # 编码观测序列
        obs_flat = obs_hist.reshape(-1, self.num_props)
        encoded_obs = self.encoder(obs_flat)
        encoded_obs = encoded_obs.reshape(batch_size, seq_len, self.encoder_output_dim)
        
        # 根据模式选择处理方式
        batch_mode = masks is not None
        if batch_mode:
            # 训练模式：使用传入的隐藏状态
            if hidden_states is None:
                raise ValueError("Hidden states not passed to RnnActor during training")
            latents, _ = self.rnn(encoded_obs, hidden_states)
            # 这里可以添加unpad_trajectories处理，如果需要的话
        else:
            # 推理模式：使用内部维护的隐藏状态
            latents, self.hidden_states = self.rnn(encoded_obs, self.hidden_states)
        
        # 取最后一个时间步的潜在表示和观测
        last_latent = latents[:, -1, :]  # [batch, hidden_dim]
        last_obs = obs_hist[:, -1, :]    # [batch, num_props]
        
        # 拼接并通过Actor网络
        actor_input = torch.cat([last_latent, last_obs], dim=-1)
        mean = self.actor(actor_input)
        return mean

    def reset_hidden_states(self, dones=None):
        """重置指定环境的隐藏状态"""
        if self.hidden_states is not None and dones is not None:
            # dones是布尔张量，标记需要重置的环境
            self.hidden_states[:, dones, :] = 0.0

class RnnCritic(nn.Module):
    def __init__(self,
                 num_props,
                 encoder_dims,
                 critic_dims,
                 encoder_output_dim,
                 hidden_dim,
                 activation,) -> None:
        super(RnnCritic, self).__init__()
        self.rnn_encoder = RnnStateHistoryEncoder(activation_fn=activation,
                                                  input_size=num_props,
                                                  encoder_dims=encoder_dims,
                                                  hidden_size=hidden_dim,
                                                  output_size=encoder_output_dim)

        self.critic = nn.Sequential(*mlp_factory(activation=activation,
                                                 input_dims=hidden_dim + num_props,
                                                 out_dims=1,
                                                 hidden_dims=critic_dims))

    def forward(self, obs_hist):
        latents = self.rnn_encoder(obs_hist)
        critic_input = torch.cat([latents[:, -1, :], obs_hist[:, -1, :]], dim=-1)
        value = self.critic(critic_input)
        return value

 
class ActorCriticRnn(nn.Module):
    is_recurrent = False
    def __init__(self,
                    num_actions,
                    num_priv,
                    num_props,
                    num_nav_commands,
                    nav_history_len,
                    history_len,
                    critic_hidden_dims=[256, 256, 256],
                    encoder_output_dim=32,
                    activation='elu',
                    init_noise_std=1.0,
                    **kwargs):
        if kwargs:
            print("ActorCriticRnn.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticRnn, self).__init__()

        activation = get_activation(activation)
        
        self.num_actions = num_actions
        self.num_priv = num_priv
        self.num_props = num_props
        self.history_len = history_len
        self.nav_history_len = nav_history_len
        self.num_nav_commands = num_nav_commands
        self.num_obs_buf = num_props + num_nav_commands

        # Policy
        self.actor = RnnActor(num_props=num_props,
                                num_actions=num_actions,
                                encoder_dims=[128],
                                actor_dims=[512, 256, 128],
                                encoder_output_dim=encoder_output_dim,
                                hidden_dim=128,
                                activation=activation)

        # Value function
        # critic_layers = mlp_factory(activation, num_props + encoder_output_dim + 3, 1, critic_hidden_dims, last_act=False)
        # self.critic = nn.Sequential(*critic_layers)
        # self.history_encoder = StateHistoryEncoder(
        # activation, num_props, history_len, encoder_output_dim)

        self.critic = RnnCritic(num_props=num_props,
                                encoder_dims=[128],
                                critic_dims=critic_hidden_dims,
                                encoder_output_dim=encoder_output_dim,
                                hidden_dim=128,
                                activation=activation)
  

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, **kwargs):
        priv_info, observations = self.get_priv_separated(observations)
        int_hist, ext_hist, obs_buf = self.extract_obs(observations)
        observations = int_hist.view(-1, self.history_len, self.num_props)
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        priv_info, observations = self.get_priv_separated(observations)
        int_hist, ext_hist, obs_buf = self.extract_obs(observations)
        observations = int_hist.view(-1, self.history_len, self.num_props)
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, observations, **kwargs):
        priv_info, observations = self.get_priv_separated(observations)
        int_hist, ext_hist, obs_buf = self.extract_obs(observations)
        observations = int_hist.view(-1, self.history_len, self.num_props)
        value = self.critic(observations)
        # history_latent = self.infer_hist_latent(observations)
        # backbone_input = torch.cat(
        #     [obs_buf, history_latent, priv_info], dim=1)
        # value = self.critic(backbone_input)
        return value

    def infer_hist_latent(self, hist):
        return self.history_encoder(hist.view(-1, self.history_len, self.num_props))

    def extract_obs(self, observations):
        int_hist = observations[:, :self.num_props*self.history_len]
        ext_hist = observations[:, -self.num_nav_commands*self.nav_history_len:]
        prop = int_hist[:, -self.num_props:]
        ext = ext_hist[:, -self.num_nav_commands:]
        obs_buf = torch.cat([prop, ext], dim=1)
        return int_hist, ext_hist, obs_buf
    
    def get_priv_separated(self, observations):
        priv_info = observations[:, :self.num_priv]
        observations = observations[:, self.num_priv:]
        return priv_info, observations
    
    def export_onnx_model(self, onnx_dir):
        # export the actor network to onnx
        import os 
        actor_cpu = self.actor.to("cpu")
        # self.actor.eval()
        dummy_input = torch.randn(1, self.num_props*self.history_len, device="cpu")
        observations = dummy_input.view(-1, self.history_len, self.num_props)
        output_path = os.path.join(onnx_dir, f"model.onnx")
        try:
            torch.onnx.export(actor_cpu, observations, output_path, verbose=False, input_names=["input"],
                            output_names=["action"])
            print(f"onnx model has been exported in  {output_path}")
            self.actor.to("cuda:0")
        except Exception as e:
            print(f"onnx export error: {str(e)}")
            raise