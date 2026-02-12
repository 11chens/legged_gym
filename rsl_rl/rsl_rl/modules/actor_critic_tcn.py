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

import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = torch.nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = torch.nn.utils.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: [Batch, Channels, Length]
        # Return embedding of the last time step
        y = self.network(x)
        return y[:, :, -1]

class Policy(nn.Module):
    def __init__(self, encoder=None, prop_encoder=None, actor=None, num_props=None, history_len=1, num_nav_commands=0, nav_history_len=0):
        super(Policy, self).__init__()
        self.encoder = encoder
        self.prop_encoder = prop_encoder
        self.actor = actor
        self.num_props = num_props
        self.history_len = history_len
        self.num_nav_commands = num_nav_commands
        self.nav_history_len = nav_history_len

    def extract_obs(self, observations):
        # input observations structure:
        # [Batch, ..., (num_nav_commands * nav_history_len), (num_props * history_len)]
        # We index from the end to be robust against extra privileged info at the start.
        
        tcn_feature_len = self.num_nav_commands * self.nav_history_len
        mlp_feature_len = self.num_props * self.history_len
                
        # Slice MLP Part (Last segment)
        mlp_part = observations[:, -mlp_feature_len:]
        
        # Slice TCN Part (Middle segment)
        tcn_part = observations[:, :tcn_feature_len]

        # Reshape TCN input for convolution: [Batch, Channels, Length]
        if self.nav_history_len > 0:
            hist_tcn_in = tcn_part.view(-1, self.nav_history_len, self.num_nav_commands)
            hist_tcn_in = hist_tcn_in.permute(0, 2, 1)  # [Batch, num_nav_commands, nav_history_len]
        else:
             hist_tcn_in = tcn_part
             
        return hist_tcn_in, mlp_part
    
    def forward(self, observations):
        hist_tcn_in, mlp_in = self.extract_obs(observations)
        if self.encoder is not None:
            latent = self.encoder(hist_tcn_in)
        else:
            latent = hist_tcn_in.flatten(1)

        if self.prop_encoder is not None:
            prop_latent = self.prop_encoder(mlp_in)
        else:
            prop_latent = mlp_in   
        
        actor_obs = torch.cat([prop_latent, latent], dim=1)
            
        action = self.actor(actor_obs) 
        return action
    

class ActorCriticTCN(nn.Module):
    is_recurrent = False # TCN uses history window, state is in obs, no internal recurrent state to pass per step
    def __init__(self,
                num_actions,
                num_priv,
                num_props,
                num_nav_commands,
                nav_history_len,
                history_len,
                actor_hidden_dims=[256, 256, 256],
                critic_hidden_dims=[256, 256, 256],
                tcn_channels=[64, 64, 64, 64], # Channels for each level of TCN. Increased depth for larger receptive field.
                tcn_kernel_size=5, # Larger kernel size
                tcn_dropout=0.2,
                prop_encoder_dims=[128, 64],
                # prop_encoder_dims=None, # No prop encoder by default
                activation='elu',
                init_noise_std=1.0,
                **kwargs):
        if kwargs:
            print("ActorCriticTCN.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticTCN, self).__init__()

        activation_fn = get_activation(activation)

        self.num_actions = num_actions
        self.num_priv = num_priv
        self.num_props = num_props
        self.history_len = history_len
        self.nav_history_len = nav_history_len
        self.num_nav_commands = num_nav_commands
        self.use_contrastive = False
        
        # TCN Encoder
        # Input: num_nav_commands (Channels)
        # Note: num_props includes Nav commands in the environment logic but we separated them in `compute_observations` obs_buf construction.
        # But wait, num_props passed here usually refers to the size of `obs_hist_buffer`'s single step?
        # Assuming num_nav_commands is the TCN input channel width.
        
        self.encoder = TemporalConvNet(
            num_inputs=num_nav_commands,
            num_channels=tcn_channels,
            kernel_size=tcn_kernel_size,
            dropout=tcn_dropout
        )
        
        self.num_latent = tcn_channels[-1] if len(tcn_channels) > 0 else num_nav_commands

        # Prop Encoder
        # Input: num_props * history_len
        prop_input_dim = num_props * history_len
        if prop_encoder_dims is not None:
            prop_layers = []
            curr_dim = prop_input_dim
            for dim in prop_encoder_dims:
                prop_layers.append(nn.Linear(curr_dim, dim))
                prop_layers.append(activation_fn)
                curr_dim = dim
            self.prop_encoder = nn.Sequential(*prop_layers)
            self.num_prop_latent = prop_encoder_dims[-1]
        else:
            self.prop_encoder = None
            self.num_prop_latent = prop_input_dim

        # Input to Actor MLP is: Flattened Obs History (num_props * history_len) + TCN Latent
        mlp_input_dim_a = self.num_prop_latent + self.num_latent
        mlp_input_dim_c = self.num_prop_latent + self.num_latent + num_priv

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation_fn)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation_fn)
        actor_ = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation_fn)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation_fn)
        self.critic = nn.Sequential(*critic_layers)
        
        print(f"TCN Encoder: {self.encoder}")
        print(f"Prop Encoder: {self.prop_encoder}")
        print(f"Critic MLP: {self.critic}")

        self.actor = Policy(
            actor=actor_,
            encoder=self.encoder,
            prop_encoder=self.prop_encoder,
            num_props=num_props,
            history_len=history_len,
            num_nav_commands=num_nav_commands,
            nav_history_len=nav_history_len
        )
        print(f"Actor MLP: {self.actor}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

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
        hist_tcn_in, mlp_in = self.actor.extract_obs(observations)
        
        latent = self.encoder(hist_tcn_in)
        if self.prop_encoder is not None:
            prop_latent = self.prop_encoder(mlp_in)
        else:
            prop_latent = mlp_in
        observations = torch.cat([prop_latent, latent, priv_info], dim=1)
        
        value = self.critic(observations)
        return value

    def get_priv_separated(self, observations):
        priv_info = observations[:, :self.num_priv]
        observations = observations[:, self.num_priv:]
        return priv_info, observations

    def export_onnx_model(self, onnx_dir):
        import os 
        actor_cpu = self.actor.to("cpu")
        # self.actor.eval()
        tcn_feature_len = self.num_nav_commands * self.nav_history_len
        mlp_feature_len = self.num_props * self.history_len
        dummy_input = torch.randn(1, tcn_feature_len + mlp_feature_len, device="cpu")
        
        output_path = os.path.join(onnx_dir, f"model.onnx")
        try:
            torch.onnx.export(actor_cpu, dummy_input, output_path, verbose=False, input_names=["obs"],
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
