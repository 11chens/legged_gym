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

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float, yaw_quat, cart2polar
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.go2.go2_nav_config import Go2NavFlatCfg
from .legged_robot import LeggedRobot

class LeggedRobotNav(LeggedRobot):
    cfg : Go2NavFlatCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        self.debug_viz = self.cfg.debug_viz
        self.sigma = 0.5
        
    def _init_buffers(self):
        """ inherit loco vars: self.commands[vx, vy, vyaw, pitch], self.actons[joint_pos]
            add nav vars: self.nav_commands[theta, rho], self.nav_actions[vx, vy, vyaw] = self.commands[:, :3]
            update vars: self.obs_buf -> nav_polciy
        """
        super()._init_buffers()
        
        self.position_targets = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False) # (x, y, z), align quat_rotate_inverse
        self.distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.nav_commands = torch.zeros(self.num_envs, self.cfg.commands.num_nav_commands, dtype=torch.float, device=self.device, requires_grad=False)
        self.nav_commands_buffer = torch.zeros(self.num_envs, 10, self.cfg.commands.num_nav_commands, dtype=torch.float, device=self.device, requires_grad=False)
        self.nav_actions = torch.zeros(self.num_envs, self.cfg.env.num_nav_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.nav_actions_before_clip = torch.zeros(self.num_envs, self.cfg.env.num_nav_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_nav_actions = torch.zeros(self.num_envs, self.cfg.env.num_nav_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_actions = torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device, requires_grad=False)
        
        self.nav_clip_min = torch.tensor([self.cfg.commands.ranges.limit_vx[0], self.cfg.commands.ranges.limit_vy[0], self.cfg.commands.ranges.limit_vyaw[0]], dtype=torch.float, device=self.device, requires_grad=False)
        self.nav_clip_max = torch.tensor([self.cfg.commands.ranges.limit_vx[1], self.cfg.commands.ranges.limit_vy[1], self.cfg.commands.ranges.limit_vyaw[1]], dtype=torch.float, device=self.device, requires_grad=False)

        self.slr_obs_buf = torch.zeros(
                self.num_envs, 45, device=self.device, dtype=torch.float)
        self.slr_obs_hist = torch.zeros(
                self.num_envs, 10, 45, device=self.device, dtype=torch.float)  

        self._load_loco_policy()

    def _load_loco_policy(self):
        """ load loco policy, which is used to compute loco actions from nav actions
            the loco policy is a slr policy, which is trained with proprioception
        """
        
        self.slr_body = torch.jit.load('controller/np3o/body_latest.jit')
        self.slr_encoder_vel = torch.jit.load('controller/np3o/encoder_vel.jit')
        
        self.slr_body =  self.slr_body.to(self.device)
        self.slr_encoder_vel =  self.slr_encoder_vel.to(self.device)

    def _compute_actions(self, nav_actions):
        """ nav_actions (loco_cmds) -> loco_actions (self.commands)
            a hacky implementation
        """
        self.commands[:, :self.num_nav_actions] = nav_actions # vx, vy, vyaw
        props = self._compute_loco_observations()
        ang_vel = self.base_ang_vel[:, 2:] * self.obs_scales.ang_vel
        lin_vel_pred = self.slr_encoder_vel(self.slr_obs_hist.view(self.num_envs, -1))
        actor_obs = torch.cat(
            (lin_vel_pred, props, ang_vel), dim=-1)
        loco_actions = self.slr_body(actor_obs)
            
        return loco_actions
    
    def _compute_loco_observations(self):
        """ It is only used for computing loco actions, NOT for updating rl agent.
        """
        # TODO: add pitch degree to self.commands
        props = torch.cat((
                self.base_ang_vel * self.obs_scales.ang_vel, # 3
                self.projected_gravity, # 3
                self.commands * self.commands_scale,
                self.reindex((self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos),
                self.reindex(self.dof_vel * self.obs_scales.dof_vel),
                self.last_dof_actions),dim=-1)
        
        self.slr_obs_hist = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([props] * self.slr_obs_hist.shape[1], dim=1),
            torch.cat([
                self.slr_obs_hist[:, 1:],
                props.unsqueeze(1)
            ], dim=1)
        )  

        return props
    
    def reset_idx(self, env_ids):
        """ origin: update terrain_cur(env_origin), cmd_curr, dofs, root_state(pos, vel), resample_cmds, fill extras
            new: remove cmd_curr, and some useless vars, update only when time out
        """
        
        if len(env_ids) == 0:
            return
        # self._update_terrain_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids) # root_state = env_origin, init_state
        self._resample_commands(env_ids)

        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_root_vel[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.slr_obs_hist[env_ids, :, :] = 0.
        self.nav_commands_buffer[env_ids, :, :] = 0.

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
         
           
    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        if not self.init_done:
            return
        move_up = self.distance < 0.2
        self.terrain_levels[env_ids] += 1 * move_up 
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    
    def reindex(self,tensor):
        """ sim2real purpose
        """
        return tensor[:,[3,4,5,0,1,2,9,10,11,6,7,8]]

    def step(self, nav_actions):
        """ origin: loco policy (dim12) -> act2tq
            new: nav policy (dim 3): loco_cmds -> act (dim 12)
        """
        # _compute_actions: loco_cmds -> loco_actions
        self.nav_actions_before_clip = nav_actions.to(self.device)
        self.nav_actions = torch.clip(self.nav_actions_before_clip, min=self.nav_clip_min, max=self.nav_clip_max)
        actions = self._compute_actions(self.nav_actions)
        self.last_dof_actions = actions
        actions = self.reindex(actions)
        
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def post_physics_step(self):
        """ Retain the original code
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback() # [new] update nav_commands

        # compute observations, rewards, resets, ...
        self.check_termination() # time out or collision
        self.compute_reward() # [new] add goal-reaching reward
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids) # [new] terrain_cur(env_origin), reset robot, resample_cmds
        self.compute_observations() # [new] update nav_commands for nav policy

        self.last_actions[:] = self.actions[:]
        self.last_nav_actions[:] = self.nav_actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()


    def _post_physics_step_callback(self):
        """ origin: resample_cmds[env_ids] at resampling_time
            new: update nav_commands every step
        """
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids, on_the_way=True)

        # pos in world -> pos in robot
        pos_diff = self.position_targets - self.root_states[:, 0:3]
        self.goal_xy_base = quat_rotate_inverse(yaw_quat(self.base_quat[:]), pos_diff)[:, :2] 
        self.nav_commands = cart2polar(self.goal_xy_base) # theta, rho

        # update distance and reach_goal
        self.distance = torch.norm(self.root_states[:, :2] - self.position_targets[:, :2], dim=1)
        self.reach_goal = self.distance < self.sigma

        self.nav_commands_buffer = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([self.nav_commands] * self.nav_commands_buffer.shape[1], dim=1),
            torch.cat([
                self.nav_commands_buffer[:, 1:],
                self.nav_commands.unsqueeze(1)
            ], dim=1)
        )  

    def _resample_commands(self, env_ids, on_the_way=False):
        """ Only resample in reset (time out)
        """
        if len(env_ids) > 0:
            if on_the_way:
                # simulate a tracking jitter
                sim_move_pos = torch_rand_float(-1.0, 1.0, (len(env_ids), 2), device=self.device)
                self.position_targets[env_ids, :2] += sim_move_pos
            else:
                _target_pos1 = torch_rand_float(-5.0, 5.0, (len(env_ids), 1), device=self.device)
                _target_pos2 = torch_rand_float(-5.0, 5.0, (len(env_ids), 1), device=self.device)
                self.position_targets[env_ids, 0:1] = self.env_origins[env_ids, 0:1] + _target_pos1
                self.position_targets[env_ids, 1:2] = self.env_origins[env_ids, 1:2] + _target_pos2

    def compute_observations(self):
        """ Computes observations for updating nav agent
        """
        nav_cmds_delay = self.nav_commands_buffer[:, -int(self.cfg.commands.delay_time / self.dt), :]

        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    nav_cmds_delay, # goal_position (theta, rho)
                                    self.nav_actions # last nav_action 
                                    ),dim=-1)
        
    def _draw_debug_vis(self):
        """ Draw position targets
        """
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_blue = gymutil.WireframeSphereGeometry(0.10, 8, 8, None, color=(0, 0, 1))
        for i in range(self.num_envs):
            x = self.position_targets[i, 0]
            y = self.position_targets[i, 1]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, 0), r=None)
            gymutil.draw_lines(sphere_blue, self.gym, self.viewer, self.envs[i], sphere_pose) 


    #### rewards

    def _reward_reach_target(self):
        return (1. /(1. + 10*torch.square(self.distance))) * self.reach_goal
    
    def _reward_stand_still(self):
        actions_norm = torch.norm(torch.abs(self.nav_actions), dim=-1) 
        return (1. /(1. + 10*torch.square(actions_norm))) * (self.distance < 0.2)
    
    def _reward_backward(self): 
        return (self.base_lin_vel[:, 0] < -0.2) * (~self.reach_goal)
    
    def _reward_nav_action_rate(self):
        return torch.square(torch.norm(self.nav_actions - self.last_nav_actions, dim=-1))

    def _reward_nav_action_limit(self):
        return torch.square(torch.norm(self.nav_actions - self.nav_actions_before_clip, dim=-1))
    
    # TODO: def FOV, penalize the goal position if it is not in the FOV
    def _reward_fov_missing(self):
        """ Reward for missing the goal position in the FOV
        """
        pass



