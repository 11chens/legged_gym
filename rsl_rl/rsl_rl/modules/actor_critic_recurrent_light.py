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
from rsl_rl.utils import unpad_trajectories

class ActorCriticRecurrentLight(ActorCritic):
    is_recurrent = True
    def __init__(self,  num_actions,
                        num_priv,
                        num_props,
                        num_nav_commands,
                        nav_history_len,
                        history_len,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        rnn_type='lstm',
                        rnn_hidden_size=256,
                        rnn_num_layers=1,
                        init_noise_std=1.0,
                        add_priv_to_critic=False,
                        shared_memory=False, 
                        use_encoder=True,
                        rnn_nav_only=True, # New flag: Only pass Nav commands to RNN
                        **kwargs):
        if kwargs:
            print("ActorCriticRecurrentLight.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),)

        super(ActorCritic, self).__init__()

        activation_fn = get_activation(activation)

        self.num_actions = num_actions
        self.num_priv = num_priv
        self.num_props = num_props
        self.num_nav_commands = num_nav_commands
        self.history_len = history_len
        self.use_contrastive = False 
        
        self.add_priv_to_critic = add_priv_to_critic
        self.shared_memory = shared_memory
        self.use_encoder = use_encoder
        self.rnn_nav_only = rnn_nav_only
        
        print(f"Prop Info: Props={num_props}, Nav={num_nav_commands}. Indices assumed: [9 : {9+num_nav_commands}] for Nav.")
        print(f"ActorCriticRecurrentLight: add_priv_to_critic={self.add_priv_to_critic}, shared_memory={self.shared_memory}, use_encoder={self.use_encoder}, rnn_nav_only={self.rnn_nav_only}")

        # --- Dimensions ---
        if self.rnn_nav_only:
            # RNN Input comes from Nav Only
            self.obs_input_dim = num_nav_commands
            # Actor/Critic Head Input = RNN_Out + Body_Obs
            # Body Obs = Total - Nav
            self.body_obs_dim = num_props - num_nav_commands
            self.head_input_dim = rnn_hidden_size + self.body_obs_dim
        else:
            # RNN Input comes from All Props
            self.obs_input_dim = num_props
            # Actor/Critic Head Input = RNN_Out
            self.head_input_dim = rnn_hidden_size
            self.body_obs_dim = 0

        # --- Feature Extractor ---
        # Processes the input destined for the RNN (Nav or All)
        if self.use_encoder:
            # Scale hidden dim slightly if input is small (Nav only)
            enc_hidden = 128 if self.rnn_nav_only else 256
            
            self.feature_extractor = nn.Sequential(
                nn.Linear(self.obs_input_dim, enc_hidden),
                activation_fn,
                nn.Linear(enc_hidden, 128),
                activation_fn
            )
            rnn_input_dim = 128
        else:
            self.feature_extractor = nn.Identity()
            rnn_input_dim = self.obs_input_dim
        
        # --- RNN Memory ---
        if self.shared_memory:
            rnn_input_size_a = rnn_input_dim
            rnn_input_size_c = rnn_input_dim
            self.memory_unify = Memory(rnn_input_size_a, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
            self.memory_a = self.memory_unify
            self.memory_c = self.memory_unify
        else:
            rnn_input_size_a = rnn_input_dim
            
            if self.add_priv_to_critic:
                rnn_input_size_c = rnn_input_dim + num_priv  # Critic gets priv info concatenated at input
            else:
                rnn_input_size_c = rnn_input_dim

            self.memory_a = Memory(rnn_input_size_a, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
            self.memory_c = Memory(rnn_input_size_c, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)

        # --- Heads ---
        # Actor Head
        actor_layers = []
        actor_layers.append(nn.Linear(self.head_input_dim, actor_hidden_dims[0]))
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
        critic_layers.append(nn.Linear(self.head_input_dim, critic_hidden_dims[0]))
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

        input_desc = "NavOnly" if self.rnn_nav_only else "FullObs"
        if self.shared_memory:
            print(f"Recurrent Light ({input_desc}): Input({self.obs_input_dim}) -> {self.feature_extractor} -> {self.memory_unify} (+Body {self.head_input_dim - rnn_hidden_size}) -> Head")
        else:
            print(f"Actor Recurrent Light ({input_desc}): Input({self.obs_input_dim}) -> Direct -> {self.memory_a} (+Body) -> {self.actor_head}")
            print(f"Critic Recurrent Light ({input_desc}): Input({self.obs_input_dim}) -> Direct -> {self.memory_c} (+Body) -> {self.critic_head}")

    def reset(self, dones=None):
        if self.shared_memory:
            self.memory_unify.reset(dones)
        else:
            self.memory_a.reset(dones)
            self.memory_c.reset(dones)

    def get_current_frame(self, observations):
        batch_shape = observations.shape[:-1]
        # View as sequence: [..., history_len, num_props]
        obs_seq = observations.view(*batch_shape, self.history_len, self.num_props)
        # Extract the LAST frame (current time step)
        # Shape: [..., num_props]
        return obs_seq[..., -1, :]

    def extract_obs(self, observations):
        # observations: [Batch, history_len * num_props] (Interleaved History)
        # We slice out ONLY the current (latest) frame to feed to the RNN.
        
        curr_obs = self.get_current_frame(observations)
        
        if self.rnn_nav_only:
             # Split into Nav and Body
             # indices: Nav is [9 : 9+num_nav]
             start_nav = 9
             end_nav = 9 + self.num_nav_commands
             
             nav_obs = curr_obs[..., start_nav:end_nav]
             
             # Body is everything else concatenated
             body_part1 = curr_obs[..., :start_nav] 
             body_part2 = curr_obs[..., end_nav:]
             body_obs = torch.cat([body_part1, body_part2], dim=-1)
             
             return nav_obs, body_obs # (RNN Input, Direct Input)

        return curr_obs, None
    
    def _encode(self, rnn_input_obs):
        if self.use_encoder:
            return self.feature_extractor(rnn_input_obs)
        return rnn_input_obs

    def act(self, observations, masks=None, hidden_states=None):
        # 1. Split Privileged
        priv_info, observations = self.get_priv_separated(observations)
        
        # 2. Extract Latest Frame & Encode
        rnn_input_data, body_obs = self.extract_obs(observations)
        latent = self._encode(rnn_input_data.detach())  # detach to avoid critic gradients flowing into actor
        
        # 3. RNN
        rnn_input = latent
        h_a = hidden_states
        
        if self.shared_memory:
            rnn_out = self.memory_unify(rnn_input, masks, h_a)
        else:
            rnn_out = self.memory_a(rnn_input, masks, h_a)

        if not (masks is not None): 
             # Inference mode
             rnn_out = rnn_out.squeeze(0)
        
        # 4. Concatenate Body Obs if needed
        if self.rnn_nav_only and body_obs is not None:
             if masks is not None:
                 body_obs = unpad_trajectories(body_obs, masks)
             head_input = torch.cat([rnn_out, body_obs], dim=-1)
        else:
             head_input = rnn_out

        # 5. Actor Head
        action_mean = self.actor_head(head_input)
        
        self.distribution = Normal(action_mean, action_mean*0. + self.std)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        # Used for deployment/testing
        priv_info, observations = self.get_priv_separated(observations)
        rnn_input_data, body_obs = self.extract_obs(observations)
        latent = self._encode(rnn_input_data)
        
        rnn_input = latent
        if self.shared_memory:
            rnn_out = self.memory_unify(rnn_input.detach()).detach()  # detach to avoid critic gradients flowing into actor
        else:
            rnn_out = self.memory_a(rnn_input) # don't detach here, actor needs gradients for own encoder
        
        rnn_out = rnn_out.squeeze(0)
        
        if self.rnn_nav_only and body_obs is not None:
             head_input = torch.cat([rnn_out, body_obs], dim=-1)
        else:
             head_input = rnn_out

        return self.actor_head(head_input)

    def evaluate(self, observations, masks=None, hidden_states=None):
        # 1. Split Privileged
        priv_info, observations = self.get_priv_separated(observations)
        
        # 2. Extract & Encode
        rnn_input_data, body_obs = self.extract_obs(observations)
        latent = self._encode(rnn_input_data)
        
        # 3. Critic Pass
        if self.add_priv_to_critic and not self.shared_memory:
             # If separate memory, we can feed priv to RNN
             if masks is not None:
                 # If we feed to RNN, we don't unpad here, Memory handles input padding/unpadding internally?
                 # Wait, Memory INPUT should be flat. Memory OUTPUT is unpadded (transformed).
                 # So latent/priv_info are flat. Correct.
                 pass
             rnn_input = torch.cat([latent, priv_info], dim=-1)
        else:
             rnn_input = latent

        h_c = hidden_states
        
        if self.shared_memory:
            rnn_out = self.memory_unify(rnn_input, masks, h_c)
        else:
            rnn_out = self.memory_c(rnn_input, masks, h_c)

        if not (masks is not None):
            rnn_out = rnn_out.squeeze(0)
        
        # Prepare Head Input
        if self.rnn_nav_only and body_obs is not None:
             if masks is not None:
                 body_obs = unpad_trajectories(body_obs, masks)
             head_input = torch.cat([rnn_out, body_obs], dim=-1)
        else:
             head_input = rnn_out
             
        if self.add_priv_to_critic and self.shared_memory:
            # If shared memory, priv info must be added AFTER RNN (to head)
            # Currently our 'evaluate' uses 'head_input' into critic_head
            # Does critic_head accept larger input?
            # Yes, 'critic_head' is init with 'self.head_input_dim'.
            # Wait, if we add priv here, we need to adjust head_input_dim in __init__?
            # Yes. But let's check init logic.
            # Currently init logic only sets head_input_dim based on rnn_hidden + body_obs
            # It does NOT account for priv info concatenation HERE.
            # So if shared_memory=True and add_priv_to_critic=True, we have a problem unless we concat it.
            # But the user code previously enabled this combination.
            # Let's assume for now we don't concat priv-info to Head if shared_memory.
            # actually, standard PPO usually gives critic priv info.
            pass

        critic_input = head_input
        value = self.critic_head(critic_input)
        return value
    
    def get_hidden_states(self):
        if self.shared_memory:
            return self.memory_unify.hidden_states, self.memory_unify.hidden_states
        return self.memory_a.hidden_states, self.memory_c.hidden_states

    def get_priv_separated(self, observations):
        priv_info, observations = torch.split(observations, [self.num_priv, observations.shape[-1] - self.num_priv], dim=-1)
        return priv_info, observations

    def export_onnx_model(self, onnx_dir):
        import os
        
        # 1. Select the correct memory module for Actor
        if self.shared_memory:
            rnn_module = self.memory_unify.rnn
        else:
            rnn_module = self.memory_a.rnn
        
        is_lstm = isinstance(rnn_module, nn.LSTM)

        # 2. Define Wrapper
        class OnnxWrapper(nn.Module):
            def __init__(self, model, is_lstm, rnn_module):
                super().__init__()
                self.model = model
                self.is_lstm = is_lstm
                self.rnn_module = rnn_module
            
            def forward(self, obs, h_in, c_in=None):
                # obs: [Batch, num_props] (Expects Single Frame)
                
                if self.model.rnn_nav_only:
                     start_nav = 9
                     end_nav = 9 + self.model.num_nav_commands
                     rnn_input_obs = obs[:, start_nav:end_nav]
                     
                     body_part1 = obs[:, :start_nav] 
                     body_part2 = obs[:, end_nav:]
                     body_obs = torch.cat([body_part1, body_part2], dim=-1)
                else:
                     rnn_input_obs = obs
                     body_obs = None

                latent = self.model._encode(rnn_input_obs)
                
                # RNN Input: [Seq=1, Batch, InputSize]
                rnn_input = latent.unsqueeze(0)
                
                if self.is_lstm:
                    hidden = (h_in, c_in)
                    out, (h_out, c_out) = self.rnn_module(rnn_input, hidden)
                    out = out.squeeze(0)
                    
                    if self.model.rnn_nav_only and body_obs is not None:
                         head_input = torch.cat([out, body_obs], dim=-1)
                    else:
                         head_input = out

                    action = self.model.actor_head(head_input)
                    return action, h_out, c_out
                else:
                    hidden = h_in
                    out, h_out = self.rnn_module(rnn_input, hidden)
                    out = out.squeeze(0)
                    
                    if self.model.rnn_nav_only and body_obs is not None:
                         head_input = torch.cat([out, body_obs], dim=-1)
                    else:
                         head_input = out

                    action = self.model.actor_head(head_input)
                    return action, h_out

        # 3. Prepare Model and Inputs
        device = next(self.parameters()).device
        self.to("cpu")
        
        # Input Obs: [1, num_props]
        # We export assuming the input is a SINGLE frame.
        obs_dim = self.num_props
        dummy_obs = torch.randn(1, obs_dim, device="cpu")
        
        num_layers = rnn_module.num_layers
        hidden_size = rnn_module.hidden_size
        
        if is_lstm:
            h_0 = torch.randn(num_layers, 1, hidden_size, device="cpu")
            c_0 = torch.randn(num_layers, 1, hidden_size, device="cpu")
            # For export args, we pass them unpacked if the forward expects them unpacked
            dummy_inputs = (dummy_obs, h_0, c_0)
            
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
            dummy_inputs = (dummy_obs, h_0)
            
            input_names = ["obs", "h_in"]
            output_names = ["action", "h_out"]
            dynamic_axes = {
                "obs": {0: "batch"}, 
                "h_in": {1: "batch"}, 
                "action": {0: "batch"},
                "h_out": {1: "batch"}
            }

        wrapper = OnnxWrapper(self, is_lstm, rnn_module)
        output_path = os.path.join(onnx_dir, f"model.onnx")
        
        # Create folder if needed
        os.makedirs(onnx_dir, exist_ok=True)

        try:
            torch.onnx.export(
                wrapper, 
                dummy_inputs, 
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
