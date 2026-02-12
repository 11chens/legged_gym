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
from .actor_critic import ActorCritic, get_activation
from .actor_critic_recurrent import Memory
from .actor_critic_encoder import FusionEncoder
from rsl_rl.utils import unpad_trajectories


class ActorCriticRecurrentEncoder(ActorCritic):
    is_recurrent = True
    def __init__(self,  num_actions,
                        num_priv,
                        num_props,
                        num_nav_commands,
                        nav_history_len,
                        history_len,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        encoder_int_hidden_dims=[256, 128, 64],
                        encoder_ext_hidden_dims=[256, 128, 64],
                        activation='elu',
                        rnn_type='lstm',
                        rnn_hidden_size=256,
                        rnn_num_layers=1,
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("ActorCriticRecurrentEncoder.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),)

        super(ActorCritic, self).__init__() # Initialize nn.Module

        activation_fn = get_activation(activation)

        self.num_actions = num_actions
        self.num_priv = num_priv
        self.num_props = num_props
        self.history_len = history_len
        self.nav_history_len = nav_history_len
        self.num_nav_commands = num_nav_commands
        self.use_contrastive = False # True to enable contrastive learning
        
        self.num_latent_int = 128
        self.num_latent_ext = 128
        self.num_latent_unify = 128

        # rnn_hidden_size = 512 # Removed hardcoded value

        encoder_hidden_dims = [512, 256]
        # encoder_hidden_dims = [512]

        # --- Encoders (Shared) ---
        # Combined Encoder
        # Use full history window as input to get explicit short-term dynamics (velocity)
        if not self.use_contrastive:
            mlp_input_dim_e = num_props * history_len
            encoder_layers = []
            encoder_layers.append(nn.Linear(mlp_input_dim_e, encoder_hidden_dims[0]))
            encoder_layers.append(activation_fn)
            for l in range(len(encoder_hidden_dims)):
                if l == len(encoder_hidden_dims) - 1:
                    encoder_layers.append(nn.Linear(encoder_hidden_dims[l], self.num_latent_unify))
                else:
                    encoder_layers.append(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l + 1]))
                    encoder_layers.append(activation_fn)
            self.encoder = nn.Sequential(*encoder_layers)
        else:
            # Use FusionEncoder instead of simple MLP
            self.encoder = FusionEncoder(num_props, history_len, num_nav_commands, hidden_dim=self.num_latent_unify)

            #  [Contrastive Learning] Projection Head
            # h (Dim Latent=128) -> MLP -> z (Dim Latent=128)
            self.projector = nn.Sequential(
                nn.Linear(self.num_latent_unify, self.num_latent_unify),
                activation_fn,
                nn.Linear(self.num_latent_unify, self.num_latent_unify)
            )
            print(f"Projector MLP: {self.projector}")

        # --- RNN Memory ---
        rnn_input_size_a = self.num_latent_unify
        # Critic gets priv info too
        rnn_input_size_c = self.num_latent_unify

        self.memory_a = Memory(rnn_input_size_a, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        self.memory_c = Memory(rnn_input_size_c, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)

        # --- Heads ---
        # Actor Head
        actor_layers = []
        actor_layers.append(nn.Linear(rnn_hidden_size, actor_hidden_dims[0]))
        actor_layers.append(activation_fn)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation_fn)
        self.actor_head = nn.Sequential(*actor_layers)

        # Critic Head
        critic_layers = []
        critic_layers.append(nn.Linear(rnn_hidden_size, critic_hidden_dims[0]))
        critic_layers.append(activation_fn)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation_fn)
        self.critic_head = nn.Sequential(*critic_layers)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        print(f"Actor Recurrent Encoder: {self.encoder} (shared)-> {self.memory_a} -> {self.actor_head}")
        print(f"Critic Recurrent Encoder: {self.encoder} (shared)-> {self.memory_c} -> {self.critic_head}")

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def extract_obs(self, observations):
        # observations: [Batch, history_len * num_props] (Interleaved History)
        # We need to extract the Navigation/External part from the interleaved sequence.
        
        # 1. Capture batch dimensions
        batch_shape = observations.shape[:-1]
        
        # 2. View as sequence: [..., history_len, num_props]
        obs_seq = observations.view(*batch_shape, self.history_len, self.num_props)
        
        # 3. Extract Nav Commands
        # Assumes Nav is at indices [9 : 9+num_nav] within num_props
        start_nav = 9
        end_nav = 9 + self.num_nav_commands
        
        nav_seq = obs_seq[..., start_nav:end_nav] # [..., Hist, NumNav]
        
        # 4. Return
        # current_prop: Full history (FusionEncoder will slice out Nav internally)
        # current_nav: Flattened Nav history [..., Hist * NumNav]
        current_prop = observations
        current_nav = nav_seq.reshape(*batch_shape, -1)
        
        return current_prop, current_nav
    
    def _encode(self, current_prop, current_nav):
        if self.use_contrastive:
            # Use Fusion Encoder
            latent = self.encoder(current_prop, current_nav)
        else:
            latent = self.encoder(current_prop)
        return latent

    def encode_projection(self, hist):
        """ Returns the projected latent z (for loss) from history.
            Flow: hist -> Encoder -> h -> Projector -> z
        """
        # hist contains both prop and nav flattened
        current_prop, current_nav = self.extract_obs(hist)
        h = self.encoder(current_prop, current_nav)
        z = self.projector(h)
        return z

    def act(self, observations, masks=None, hidden_states=None):
        # 1. Split Privileged
        priv_info, observations = self.get_priv_separated(observations)
        
        # 2. Extract & Encode
        current_prop, current_nav = self.extract_obs(observations)
        latent = self._encode(current_prop, current_nav)
        
        # 3. Prepare RNN Input
        # Removed .detach() to allow encoder training from Actor gradients
        rnn_input = latent
        
        # 4. Run RNN
        # hidden_states is h_a (passed from PPO)
        h_a = hidden_states
        rnn_out = self.memory_a(rnn_input, masks, h_a) 
        if not (masks is not None): 
             # Inference mode (Memory returned [1, Batch, Hidden])
             rnn_out = rnn_out.squeeze(0)
        
        # 5. Actor Head
        action_mean = self.actor_head(rnn_out)
        
        self.distribution = Normal(action_mean, action_mean*0. + self.std)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        # Used for deployment/testing
        priv_info, observations = self.get_priv_separated(observations)
        current_prop, current_nav = self.extract_obs(observations)
        latent = self._encode(current_prop, current_nav)
        
        rnn_input = latent
        rnn_out = self.memory_a(rnn_input)
        rnn_out = rnn_out.squeeze(0)
        
        return self.actor_head(rnn_out)

    def evaluate(self, observations, masks=None, hidden_states=None):
        # 1. Split Privileged
        priv_info, observations = self.get_priv_separated(observations)
        
        # 2. Extract & Encode
        current_prop, current_nav = self.extract_obs(observations)
        latent = self._encode(current_prop, current_nav)
        
        # 3. Predict Value
        rnn_input = latent
        h_c = hidden_states
        
        # Memory returns flattened trajectories in training mode (masks is not None)
        rnn_out = self.memory_c(rnn_input, masks, h_c)
        if not (masks is not None):
             rnn_out = rnn_out.squeeze(0)

        # 4. Concatenate Privileged Info
        # [Note]: priv_info needs to be unpadded/matched to rnn_out if unpadding happened inside memory
        # 'Memory' calls unpad_trajectories(out, masks)
        # So we must unpad 'priv_info' as well!
        
        # if masks is not None:
        #      priv_info = unpad_trajectories(priv_info, masks)

        # critic_input = torch.cat([priv_info, rnn_out], dim=-1)
        critic_input = rnn_out
        value = self.critic_head(critic_input)
        return value
    
    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states

    def get_priv_separated(self, observations):
        priv_info, observations = torch.split(observations, [self.num_priv, observations.shape[-1] - self.num_priv], dim=-1)
        return priv_info, observations

    def export_onnx_model(self, onnx_dir):
        import os
        
        # 1. Define Wrapper
        class OnnxWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, current_prop, hidden_states):
                # Note: observations should be without privileged info
                latent = self.model._encode(current_prop)
                
                rnn_input = latent.unsqueeze(0)
                # Handle LSTM tuple vs GRU tensor
                if isinstance(hidden_states, tuple):
                    out, new_hidden_states = self.model.memory_a.rnn(rnn_input, hidden_states)
                else:
                    out, new_hidden_states = self.model.memory_a.rnn(rnn_input, hidden_states)
                
                out = out.squeeze(0)
                action = self.model.actor_head(out)
                return action, new_hidden_states

        # 2. Prepare Model and Inputs
        device = next(self.parameters()).device
        self.to("cpu")
        
        # Input Obs
        obs_dim = self.num_props * self.history_len
        dummy_obs = torch.randn(1, obs_dim, device="cpu")
        
        # Input Hidden
        num_layers = self.memory_a.rnn.num_layers
        hidden_size = self.memory_a.rnn.hidden_size
        
        is_lstm = isinstance(self.memory_a.rnn, nn.LSTM)
        if is_lstm:
            h_0 = torch.randn(num_layers, 1, hidden_size, device="cpu")
            c_0 = torch.randn(num_layers, 1, hidden_size, device="cpu")
            dummy_hidden = (h_0, c_0)
            input_names = ["obs", "h_in", "c_in"]
            output_names = ["action", "h_out", "c_out"]
            dynamic_axes = {
                "obs": {0: "batch"}, 
                "h_in": {1: "batch"}, 
                "c_in": {1: "batch"},
                "action": {0: "batch"},
                "h_out": {1: "batch"},
                "c_out": {1: "batch"}
            }
        else: # GRU
            h_0 = torch.randn(num_layers, 1, hidden_size, device="cpu")
            dummy_hidden = h_0
            input_names = ["obs", "h_in"]
            output_names = ["action", "h_out"]
            dynamic_axes = {
                "obs": {0: "batch"}, 
                "h_in": {1: "batch"}, 
                "action": {0: "batch"},
                "h_out": {1: "batch"}
            }

        wrapper = OnnxWrapper(self)
        output_path = os.path.join(onnx_dir, "model.onnx")
        
        try:
            torch.onnx.export(
                wrapper, 
                (dummy_obs, dummy_hidden), 
                output_path, 
                verbose=False, 
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=11
            )
            print(f"ONNX model exported to {output_path}")
        except Exception as e:
            print(f"ONNX export error: {e}")
        finally:
            self.to(device)
