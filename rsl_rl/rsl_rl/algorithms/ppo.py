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
import torch.optim as optim

# from rsl_rl.modules import ActorCriticEncoder
from rsl_rl.storage import RolloutStorage
import torch.nn.functional as F

class PPO:
    # actor_critic: ActorCriticEncoder
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.enale_contrastive_learning = self.actor_critic.use_contrastive

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        # obs = critic_obs
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def process_env_step(self, next_obs, rewards, dones, infos):
        self.transition.next_observations = next_obs
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def compute_lipschitz_loss(self, current_states, next_states, orig_actions, orig_values, masks=None, hidden_states=None):
        batch_size = current_states.size(0)
        u = torch.rand(batch_size, 1, device=current_states.device)
        
        # Interpolate between current and next states (s_t -> s_{t+1})
        interp_states = current_states + u * (next_states - current_states)
        
        # Compute outputs for interpolated states (pass hidden_states in RNN mode)
        if self.actor_critic.is_recurrent:
            self.actor_critic.act(interp_states, masks=masks, hidden_states=hidden_states[0])
            interp_actions = self.actor_critic.action_mean
            interp_values = self.actor_critic.evaluate(interp_states, masks=masks, hidden_states=hidden_states[1])
        else:
            self.actor_critic.act(interp_states)
            interp_actions = self.actor_critic.action_mean
            interp_values = self.actor_critic.evaluate(interp_states)
        
        # Compute Lipschitz smoothness loss
        actor_smoothness = F.mse_loss(interp_actions, orig_actions)
        critic_smoothness = F.mse_loss(interp_values, orig_values)
        
        return actor_smoothness, critic_smoothness

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_imitation_loss = 0
        mean_contrastive_loss = 0
        contrastive_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, next_obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:


                self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate


                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                # # Lipschitz Smoothness Loss
                # smooth_actor, smooth_critic = self.compute_lipschitz_loss(
                #     obs_batch, next_obs_batch, mu_batch, value_batch, 
                #     masks=masks_batch, hidden_states=hid_states_batch
                # )

                clip_mins = torch.tensor([-0.5, -0.5, -0.5, -0.5], device=mu_batch.device)
                clip_maxs = torch.tensor([0.5,  0.5,  0.5,  0.5], device=mu_batch.device)
                regularization_loss = torch.mean(
                    torch.sum((mu_batch - torch.clip(mu_batch, min=clip_mins, max=clip_maxs))**2, dim=-1)
                )

                # [Contrastive Learning Update]
                # Expects obs_batch to have Task ID as the last element [..., label]
                if self.enale_contrastive_learning:
                    # 1. Extract Labels (Task ID is at index 0, in priv_info)
                    # Handle both 2D [Batch, Dim] and 3D [Batch, Time, Dim] tensors
                    task_labels = obs_batch[..., 0].long().flatten()
                    
                    # 2. Extract Encoder Input (Remove Privileged Info)
                    # Use the module's own method to ensure consistency
                    _, actor_obs_batch = self.actor_critic.get_priv_separated(obs_batch)
                    
                    # 3. Get Projected Embeddings z (via Projector)
                    # Flow: Obs -> Encoder -> h -> Projector -> z
                    z = self.actor_critic.encode_projection(actor_obs_batch)
                    
                    if z is not None:
                        # Flatten z for contrastive computation: [N, Latent]
                        z_flat = z.reshape(-1, z.shape[-1])

                        # 4. Compute Prototype-based Contrastive Loss (Memory Efficient)
                        # Instead of O(N^2) pairwise matrix, we use O(N*K) prototypes.
                        z_norm = F.normalize(z_flat, dim=1)
                        
                        # Find unique task labels present in this batch
                        unique_labels, inverse_indices = torch.unique(task_labels, sorted=True, return_inverse=True)
                        
                        if len(unique_labels) > 1:
                            centroids = []
                            # Compute centroid for each class present
                            for i in range(len(unique_labels)):
                                # Samples belonging to the i-th unique label
                                mask = (inverse_indices == i)
                                center = z_norm[mask].mean(dim=0)
                                centroids.append(F.normalize(center, dim=0))
                            
                            # Stack centroids: [K, D] -> [3, 256]
                            centroid_stack = torch.stack(centroids)
                            
                            # Compute logits: Sim(z, centroids) -> [N, K]
                            logits = torch.matmul(z_norm, centroid_stack.T) / 0.07
                            
                            # Target is the index of the correct centroid (inverse analysis)
                            contrastive_loss = F.cross_entropy(logits, inverse_indices)
                        else:
                            # If only 1 class is present, we cannot do contrastive learning
                            contrastive_loss = 0.0

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
                loss += 0.01 * regularization_loss
                loss += 0.01 * contrastive_loss # Reduced from 0.1 to balance with PPO losses (3.5 -> 0.0035)

                # loss += 0.05 * smooth_actor + 0.01 * smooth_critic
                
                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                if self.enale_contrastive_learning:
                    mean_contrastive_loss += contrastive_loss.item()

                ######### update: MSE loss #########
                if hasattr(self.actor_critic, "estimator"):
                    P_base_targ = obs_batch[:, :3]
                    obs_batch = obs_batch[:, 3:]
                    for epoch in range(5):
                        P_base_pred = self.actor_critic.estimate(obs_batch)
                        imitation_loss = F.mse_loss(P_base_pred, P_base_targ)
                        self.optimizer.zero_grad()
                        imitation_loss.backward()
                        self.optimizer.step()

                        mean_imitation_loss += imitation_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_imitation_loss /= num_updates * 5
        mean_contrastive_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_imitation_loss, mean_contrastive_loss
