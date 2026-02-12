# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .ppo import PPO

class PPOChunk(PPO):
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

                # --- Smoothness Loss (Chunking Specific) ---
                # mu_batch: [Batch, Chunk * ActionDim]
                # We expect self.actor_critic.chunk_size to ideally be present, or we infer
                chunk_size = getattr(self.actor_critic, 'chunk_size', 1)
                
                if chunk_size > 1:
                    batch_size = mu_batch.shape[0]
                    # Reshape to [Batch, Chunk, ActionDim]
                    # ActionDim = TotalDim / Chunk
                    action_dim = mu_batch.shape[1] // chunk_size
                    mu_reshaped = mu_batch.view(batch_size, chunk_size, action_dim)
                    
                    # Compute difference between consecutive steps in the chunk
                    # diff: [Batch, Chunk-1, ActionDim]
                    diff = mu_reshaped[:, 1:, :] - mu_reshaped[:, :-1, :]
                    
                    smooth_loss = torch.mean(diff ** 2)
                else:
                    smooth_loss = 0.0


                # --- Contrastive Learning Update (Restored) ---
                if self.enale_contrastive_learning:
                    # 1. Extract Labels (Task ID is at index 0, in priv_info)
                    task_labels = obs_batch[..., 0].long().flatten()
                    
                    # 2. Extract Encoder Input (Remove Privileged Info)
                    _, actor_obs_batch = self.actor_critic.get_priv_separated(obs_batch)
                    
                    # 3. Get Projected Embeddings z (via Projector)
                    z = self.actor_critic.encode_projection(actor_obs_batch)
                    
                    if z is not None:
                        # Flatten z for contrastive computation: [N, Latent]
                        z_flat = z.reshape(-1, z.shape[-1])

                        # 4. Compute Prototype-based Contrastive Loss
                        z_norm = F.normalize(z_flat, dim=1)
                        unique_labels, inverse_indices = torch.unique(task_labels, sorted=True, return_inverse=True)
                        
                        if len(unique_labels) > 1:
                            centroids = []
                            for i in range(len(unique_labels)):
                                mask = (inverse_indices == i)
                                center = z_norm[mask].mean(dim=0)
                                centroids.append(F.normalize(center, dim=0))
                            
                            centroid_stack = torch.stack(centroids)
                            logits = torch.matmul(z_norm, centroid_stack.T) / 0.07
                            contrastive_loss = F.cross_entropy(logits, inverse_indices)
                        else:
                            contrastive_loss = 0.0

                # Regularization Loss (Penalize actions outside [-0.5, 0.5])
                # Using scalar clamp to support variable chunk sizes/action dims
                regularization_loss = torch.mean(
                    torch.sum((mu_batch - torch.clamp(mu_batch, min=-0.5, max=0.5))**2, dim=-1)
                )

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
                
                # Add Smoothness 
                loss += 0.01 * smooth_loss
                
                # Add Contrastive
                loss += 0.01 * contrastive_loss
                
                # Add Regularization from original PPO
                loss += 0.01 * regularization_loss

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                if self.enale_contrastive_learning and isinstance(contrastive_loss, torch.Tensor):
                    mean_contrastive_loss += contrastive_loss.item()
                elif self.enale_contrastive_learning:
                    mean_contrastive_loss += contrastive_loss

                ######### update: MSE loss (Restored) #########
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
