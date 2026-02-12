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
    def __init__(self, encoder=None, actor=None, num_props=None, history_len=1, nav_history_len=1, num_nav_commands=3):
        super(Policy, self).__init__()
        self.encoder = encoder
        self.actor = actor
        self.num_props = num_props
        self.history_len = history_len
        self.nav_history_len = nav_history_len
        self.num_nav_commands = num_nav_commands

    def extract_obs(self, observations):
        # 1. Capture batch dimensions
        batch_shape = observations.shape[:-1]
        
        # 2. Reshape [Batch, Len * Props] -> [Batch, Len, Props]
        # We assume observations is strictly the history buffer here
        obs_seq = observations.view(*batch_shape, self.history_len, self.num_props)
        
        # 3. Extract Nav (Indices 9 to 9+Nav)
        start_nav = 9
        end_nav = 9 + self.num_nav_commands
        nav_seq = obs_seq[..., start_nav:end_nav]
        
        # 4. Flatten External History
        ext_hist = nav_seq.reshape(*batch_shape, -1)
        
        # 5. Full History (Proprio + Nav interleaved)
        prop_hist = observations
        
        # 6. Current Step Obs (Last step)
        curr_obs = observations[..., -self.num_props:]

        return prop_hist, ext_hist, curr_obs
    
    def forward(self, observations):
        prop_hist, ext_hist, curr_obs = self.extract_obs(observations)
        if self.encoder is not None:
            latent = self.encoder(prop_hist, ext_hist)
            actor_obs = latent
        else:
            actor_obs = prop_hist
        action = self.actor(actor_obs) 
        return action
    

class FusionEncoder(nn.Module):
    """
    Splits observation history into Internal (Proprioception) and External (Sigma Points) streams,
    encodes them separately, and fuses them.
    Reference from user request:
    - External (Sigma Points): MLP[128] -> LayerNorm -> ReLU -> MLP[128] -> ReLU
    - Internal (Proprio): MLP[64] -> LayerNorm -> ReLU
    - Fusion: Linear(128+64, 32) -> h
    """
    def __init__(self, num_props, history_len, num_nav_commands, hidden_dim=32):
        super(FusionEncoder, self).__init__()
        self.num_props = num_props
        self.history_len = history_len
        self.num_nav_commands = num_nav_commands
        
        # Dimensions
        self.dim_ext = num_nav_commands * history_len
        self.dim_int = (num_props - num_nav_commands) * history_len
        
        # 1. External Encoder (Point MLP)
        self.point_mlp = nn.Sequential(
            nn.Linear(self.dim_ext, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # 2. Internal Encoder (Prop MLP)
        self.prop_mlp = nn.Sequential(
            nn.Linear(self.dim_int, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )
        
        # 3. Fusion Layer
        self.fusion_layer = nn.Linear(128 + 64, hidden_dim) 
        
    def forward(self, prop_history, ext_history):
        # prop_history: [Batch, num_props * history_len]
        # ext_history:  [Batch, num_nav * history_len] (or distinct external feats)
        
        # 1. Process External (Nav/Sigma) - Direct pass
        # Handle 3D inputs [Batch, Seq, Dim]
        batch_shape = ext_history.shape[:-1]
        
        # Flattening might be needed if input is [ Batch, Seq, Feat] -> [ Batch*Seq, Feat ]
        # But here we want to preserve Batch/Seq structure for MLPs?
        # Standard MLP in pytorch handles multidimensional inputs applied to last dim.
        h_ext = self.point_mlp(ext_history)

        # 2. Process Internal (Proprioception)
        # We need to REMOVE the Nav commands (indices 9:9+3) from prop_history 
        # to match dim_int = (num_props - num_nav) * len
        
        # Reshape to access per-step indices: [..., history_len, num_props]
        obs_seq = prop_history.view(*batch_shape, self.history_len, self.num_props)
        
        start_ext = 9
        end_ext = 9 + self.num_nav_commands
        
        # Slice out the nav/external part from internal stream
        int_seq_1 = obs_seq[..., :start_ext]
        int_seq_2 = obs_seq[..., end_ext:]
        
        int_seq = torch.cat([int_seq_1, int_seq_2], dim=-1) # [..., len, dim_int_per_step]
        
        # Flatten time dim back: [..., len * (dim_props - dim_nav)]
        int_flat = int_seq.reshape(*batch_shape, -1)
        
        h_int = self.prop_mlp(int_flat)
        
        # 3. Fuse
        h_cat = torch.cat([h_ext, h_int], dim=-1)
        h = self.fusion_layer(h_cat)
        
        return h

class ActorCriticEncoder(nn.Module):
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
                encoder_hidden_dims=[256, 128], # Ignored now
                activation='elu',
                init_noise_std=1.0,
                has_encoder=True,
                **kwargs):
        if kwargs:
            print("ActorCriticEncoder.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticEncoder, self).__init__()

        activation = get_activation(activation)

        self.num_actions = num_actions
        self.num_priv = num_priv
        self.num_props = num_props
        self.history_len = history_len
        self.nav_history_len = nav_history_len
        self.num_nav_commands = num_nav_commands
        self.num_obs_buf = num_props + num_nav_commands
        self.has_encoder = has_encoder
        self.num_latent = 32 # Updated to match Fusion Layer Output
        self.use_contrastive = True

        # mlp_input_dim_a = self.num_props + self.num_latent if self.has_encoder else num_props * history_len
        # mlp_input_dim_c = self.num_props + self.num_latent if self.has_encoder else num_props * history_len
        
        mlp_input_dim_a = self.num_latent
        mlp_input_dim_c = self.num_latent
        mlp_input_dim_c += num_priv

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

        # Encoder (Fusion)
        encoder_ = None
        if self.has_encoder:
            encoder_ = FusionEncoder(num_props, history_len, num_nav_commands, hidden_dim=self.num_latent)

        # [Contrastive Learning] Projection Head
        # h (Dim Latent=256) -> MLP -> z (Dim Latent=256)
        # Used for InfoNCE Loss calculation on z, while Actor uses h.
        self.projector = None
        if self.has_encoder:
            projector_layers = []
            # Layer 1: h -> h
            projector_layers.append(nn.Linear(self.num_latent, self.num_latent))
            projector_layers.append(activation)
            # Layer 2: h -> z (z dim = h dim)
            projector_layers.append(nn.Linear(self.num_latent, self.num_latent))
            self.projector = nn.Sequential(*projector_layers)
            print(f"Projector MLP: {self.projector}")

        self.actor = Policy(
            actor=actor_,
            encoder=encoder_ if self.has_encoder else None,
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
        prop_hist, ext_hist, curr_obs = self.actor.extract_obs(observations)
        if self.has_encoder:
            latent = self._encode_wrapper(prop_hist, ext_hist)
            # observations = torch.cat([priv_info, curr_obs, latent], dim=1)
            observations = torch.cat([priv_info, latent], dim=1)
        else:
            observations = prop_hist
        value = self.critic(observations)
        return value

    def encode(self, hist):
        # Need to extract bits from hist directly if calling this from outside?
        # Or assumes hist is already (prop, ext) tuple?
        # Given 'hist' argument name, it's likely just one tensor.
        # Let's assume it's the raw observations history buffer.
        prop_hist, ext_hist, _ = self.actor.extract_obs(hist)
        latent = self.actor.encoder(prop_hist, ext_hist)
        return latent

    # Helper because self.encode signature was `encode(hist)`
    def _encode_wrapper(self, prop_hist, ext_hist):
        return self.actor.encoder(prop_hist, ext_hist)

    def encode_projection(self, hist):
        """ Returns the projected latent z (for loss) from history.
            Flow: hist -> Encoder -> h -> Projector -> z
        """
        if self.projector is None:
            return None
        prop_hist, ext_hist, _ = self.actor.extract_obs(hist)
        h = self.actor.encoder(prop_hist, ext_hist)
        z = self.projector(h)
        return z

    def get_priv_separated(self, observations):
        priv_info = observations[:, :self.num_priv]
        observations = observations[:, self.num_priv:]
        return priv_info, observations

    def export_onnx_model(self, onnx_dir):
        import os 
        actor_cpu = self.actor.to("cpu")
        # self.actor.eval()
        dummy_input = torch.randn(1, self.num_props*self.history_len, device="cpu")
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
