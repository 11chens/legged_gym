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

class Policy(nn.Module):
    def __init__(self, encoder_int=None, encoder_ext=None, actor=None, num_props=None, history_len=1, nav_history_len=1, num_nav_commands=3):
        super(Policy, self).__init__()
        self.encoder_ext = encoder_ext
        self.encoder_int = encoder_int
        self.actor = actor
        self.num_props = num_props
        self.history_len = history_len
        self.nav_history_len = nav_history_len
        self.num_nav_commands = num_nav_commands

    def extract_obs(self, observations):
        int_hist = observations[:, :self.num_props*self.history_len]
        ext_hist = observations[:, -self.num_nav_commands*self.nav_history_len:]
        prop = int_hist[:, -self.num_props:]
        ext = ext_hist[:, -self.num_nav_commands:]
        obs_buf = torch.cat([prop, ext], dim=1)
        return int_hist, ext_hist, obs_buf
    
    def forward(self, observations):
        int_hist, ext_hist, obs_buf = self.extract_obs(observations)
        latent_int = self.encoder_int(int_hist)
        latent_ext = self.encoder_ext(ext_hist)
        # actor_obs = torch.cat([obs_buf, latent_int.detach(), latent_ext.detach()], dim=1)
        actor_obs = torch.cat([obs_buf, latent_int, latent_ext], dim=1)
        action = self.actor(actor_obs) 
        return action
    

class ActorCriticAsync(nn.Module):
    is_recurrent = False
    def __init__(self,
                num_actions,
                num_priv,
                num_props,
                num_nav_commands,
                nav_history_len,
                history_len,
                actor_hidden_dims=[256, 256, 256],
                critic_hidden_dims=[256, 256, 256],
                encoder_int_hidden_dims=[256, 128, 64],
                encoder_ext_hidden_dims=[256, 128, 64],

                # encoder_int_hidden_dims=[512, 256, 128],
                # encoder_ext_hidden_dims=[512, 256, 128],
                # encoder_hidden_dims=[256, 128, 64],
                activation='elu',
                init_noise_std=1.0,
                **kwargs):
        if kwargs:
            print("ActorCriticAsync.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticAsync, self).__init__()

        activation = get_activation(activation)

        self.num_actions = num_actions
        self.num_priv = num_priv
        self.num_props = num_props
        self.history_len = history_len
        self.nav_history_len = nav_history_len
        self.num_nav_commands = num_nav_commands
        self.num_obs_buf = num_props + num_nav_commands
        self.num_latent_int = 16
        self.num_latent_ext = 16


        mlp_input_dim_a = self.num_obs_buf + self.num_latent_int + self.num_latent_ext
        mlp_input_dim_c = self.num_obs_buf + self.num_latent_int + self.num_latent_ext + self.num_priv
        mlp_input_dim_e_int = num_props * history_len
        mlp_input_dim_e_ext = num_nav_commands * nav_history_len

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        actor_ = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)
        print(f"Critic MLP: {self.critic}")

        # encoder_int
        encoder_int_layers = []
        encoder_int_layers.append(nn.Linear(mlp_input_dim_e_int, encoder_int_hidden_dims[0]))
        encoder_int_layers.append(activation)
        for l in range(len(encoder_int_hidden_dims)):
            if l == len(encoder_int_hidden_dims) - 1:
                encoder_int_layers.append(nn.Linear(encoder_int_hidden_dims[l], self.num_latent_int))
            else:
                encoder_int_layers.append(nn.Linear(encoder_int_hidden_dims[l], encoder_int_hidden_dims[l + 1]))
                encoder_int_layers.append(activation)
        encoder_int_ = nn.Sequential(*encoder_int_layers)

        # encoder_ext
        encoder_ext_layers = []
        encoder_ext_layers.append(nn.Linear(mlp_input_dim_e_ext, encoder_ext_hidden_dims[0]))
        encoder_ext_layers.append(activation)
        for l in range(len(encoder_ext_hidden_dims)):
            if l == len(encoder_ext_hidden_dims) - 1:
                encoder_ext_layers.append(nn.Linear(encoder_ext_hidden_dims[l], self.num_latent_ext))
            else:
                encoder_ext_layers.append(nn.Linear(encoder_ext_hidden_dims[l], encoder_ext_hidden_dims[l + 1]))
                encoder_ext_layers.append(activation)
        encoder_ext_ = nn.Sequential(*encoder_ext_layers)

        self.actor = Policy(
            actor=actor_,
            encoder_int=encoder_int_,
            encoder_ext=encoder_ext_,
            num_props=num_props,
            history_len=history_len,
            nav_history_len=nav_history_len,
            num_nav_commands=num_nav_commands,
        )
        print(f"Actor MLP: {self.actor}")

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
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        priv_info, observations = self.get_priv_separated(observations)
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, observations, **kwargs):
        priv_info, observations = self.get_priv_separated(observations)
        int_hist, ext_hist, obs_buf = self.actor.extract_obs(observations)
        latent_int, latent_ext = self.encode(int_hist, ext_hist)
        observations = torch.cat([priv_info, obs_buf, latent_int, latent_ext], dim=1)
        value = self.critic(observations)
        return value

    def encode(self, int_hist, ext_hist):
        latent_int = self.actor.encoder_int(int_hist)
        latent_ext = self.actor.encoder_ext(ext_hist)
        return latent_int, latent_ext

    def get_priv_separated(self, observations):
        priv_info = observations[:, :self.num_priv]
        observations = observations[:, self.num_priv:]
        return priv_info, observations

    def export_onnx_model(self, onnx_dir):
        import os 
        actor_cpu = self.actor.to("cpu")
        # self.actor.eval()
        dummy_input = torch.randn(1, self.num_props*self.history_len + self.num_nav_commands*self.nav_history_len, device="cpu")
        output_path = os.path.join(onnx_dir, f"model.onnx")
        try:
            torch.onnx.export(actor_cpu, dummy_input, output_path, verbose=False, input_names=["input"],
                            output_names=["action"])
            print(f"onnx model has been exported in  {output_path}")
            self.actor.to("cuda:0")
        except Exception as e:
            print(f"onnx export error: {str(e)}")
            raise
    
def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
