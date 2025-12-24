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
from torch import Tensor, squeeze
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float, yaw_quat, cart2polar, quat_to_rot_matrix
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.go2.go2_nav_config import Go2NavFlatCfg
from .legged_robot import LeggedRobot
from legged_gym.utils.camera_sensor import CameraSensor
from legged_gym.perception.tracker import PCATargetTracker
from legged_gym.perception.surface_geometry import Cylinder, Cuboid, Sphere
from legged_gym.utils.visualization import VisualizationUtils

class LeggedRobotNav(LeggedRobot):
    cfg : Go2NavFlatCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        self.debug_viz = self.cfg.debug_viz
        self.sigma = 0.4
        self.camera_sensor = CameraSensor(
            batch_size=self.num_envs, 
            cfg=self.cfg.camera_sensor,
            device=self.device,
            )
    
        
        # Gripper Config
        self.gripper_width = self.cfg.gripper.gripper_width
        self.gripper_offset = torch.tensor(self.cfg.gripper.gripper_offset, device=self.device).repeat(self.num_envs, 1)

        # Initialize Perception Tracker
        target_cfg = self.cfg.target
        
        # Pre-instantiate shapes
        self.shapes = {
            "cylinder": Cylinder(self.device),
            "cuboid": Cuboid(self.device),
            "sphere": Sphere(self.device)
        }
        
        self.object_dims = torch.zeros(self.num_envs, 3, device=self.device)
        self.local_points = torch.zeros(self.num_envs, target_cfg.perception.num_sample_points, 3, device=self.device)
        self.local_normals = torch.zeros(self.num_envs, target_cfg.perception.num_sample_points, 3, device=self.device)
        
        # Object State
        self.object_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.object_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.object_quat[:, 3] = 1.0 # Identity

        self.x_axis_local = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self.y_axis_local = torch.tensor([0.0, 1.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self.z_axis_local = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)

        # Always use mixed mode
        self.shape_types_list = target_cfg.shape.types
        self.env_shape_type_indices = torch.randint(0, len(self.shape_types_list), (self.num_envs,), device=self.device)

        # Randomize props for all envs
        self._reset_object_shapes(torch.arange(self.num_envs, device=self.device))
        
        self.tracker = PCATargetTracker(
            shape_type="mixed", # We handle everything as mixed/custom now
            shape_params=torch.zeros(1, device=self.device), # Dummy
            num_envs=self.num_envs,
            device=self.device,
            num_sample_points=target_cfg.perception.num_sample_points,
            local_points=self.local_points,
            local_normals=self.local_normals,
            object_dims=self.object_dims
        )
        
        # Visualization Utils
        self.vis_utils = VisualizationUtils(self)
        
        # Sigma Points in Base Frame (for Obs)
        self.sigma_points_base = torch.zeros(self.num_envs, self.cfg.env.num_sigma_points, 3, device=self.device)

        # Cache config parameters
        self.target_cfg = self.cfg.target
        self.perception_frame = self.target_cfg.perception.frame
        self.use_geometric_weight = self.target_cfg.perception.use_geometric_weight
        self.randomize_orientation = self.target_cfg.init.randomize_orientation
        self.debug_timer = self.target_cfg.perception.debug_timer
        self.debug_info = self.target_cfg.perception.debug_info

        # Cache camera drift params
        self.enable_out_of_view_drift = self.cfg.camera_sensor.enable_out_of_view_drift
        self.drift_scale = self.cfg.camera_sensor.drift_scale
        self.max_out_of_view_duration = self.cfg.camera_sensor.max_out_of_view_duration
        self.save_debug_images = self.cfg.camera_sensor.save_debug_images

        # Pre-allocate tensors for resampling
        self.t_min_default = torch.ones((self.num_envs, 1), device=self.device) * 0.5
        self.t_max_default = torch.ones((self.num_envs, 1), device=self.device) * 1.5
        self.dir_cam_z = torch.ones((self.num_envs, 1), device=self.device)
    
        # Cache camera params as tensors for efficient indexing
        def to_tensor(val):
            if isinstance(val, (int, float)):
                return torch.full((self.num_envs, 1), val, device=self.device)
            elif isinstance(val, torch.Tensor):
                if val.dim() == 0:
                    return val.expand(self.num_envs, 1)
                return val
            return val

        self.cam_fx = to_tensor(self.camera_sensor.fx)
        self.cam_fy = to_tensor(self.camera_sensor.fy)
        self.cam_cx = to_tensor(self.camera_sensor.cx)
        self.cam_cy = to_tensor(self.camera_sensor.cy)
        self.cam_img_w = to_tensor(self.camera_sensor.img_width)
        self.cam_img_h = to_tensor(self.camera_sensor.img_height)

        # Cache camera params dicts for tracker
        self.camera_params_dict = {
            'fx': self.camera_sensor.fx,
            'fy': self.camera_sensor.fy,
            'cx': self.camera_sensor.cx,
            'cy': self.camera_sensor.cy,
            'img_width': self.camera_sensor.img_width,
            'img_height': self.camera_sensor.img_height
        }
        
        self.camera_transform_dict = {
            'R': self.camera_sensor.R,
            'T': self.camera_sensor.T
        }
        
    def _init_buffers(self):
        """ inherit loco vars: self.commands[vx, vy, vyaw, pitch], self.actons[joint_pos]
            add nav vars: self.nav_commands[theta, rho], self.nav_actions[vx, vy, vyaw] = self.commands[:, :3]
            update vars: self.obs_buf -> nav_polciy
        """
        super()._init_buffers()
                
        self.P_base = torch.zeros(self.num_envs, self.cfg.env.num_position, dtype=torch.float, device=self.device, requires_grad=False)
        self.distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.objct_z = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.object_moved = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.obj_is_vertical = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.obj_is_too_long = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.object_z_offset = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.max_dim_vals = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.max_dim_inds = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.out_of_view_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.out_of_view_drift = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        
        self.physically_out_of_view = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        
        # Success Timer for Dense Reward
        self.success_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.is_success_state = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)

        # Optimal Grasp Pose Buffers
        self.env_start_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.optimal_grasp_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.optimal_grasp_quat = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.optimal_grasp_quat[:, 3] = 1.0 # Identity
        
        # Hint Pose Buffers
        self.hint_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.hint_quat = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.hint_quat[:, 3] = 1.0 # Identity
        
        # Optimal Approach Direction (Normalized, pointing to object)
        self.optimal_approach_dir = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.optimal_approach_dir[:, 0] = 1.0

        self.gripper_world = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

        self.task_flags = torch.zeros(self.num_envs, self.cfg.env.num_task_flags, dtype=torch.float, device=self.device, requires_grad=False)
        self.is_place = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.is_pick = torch.ones(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)

        self.nav_commands = torch.zeros(self.num_envs, self.cfg.commands.num_nav_commands, dtype=torch.float, device=self.device, requires_grad=False)

        self.nav_actions = torch.zeros(self.num_envs, self.cfg.env.num_nav_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.nav_actions_before_clip = torch.zeros(self.num_envs, self.cfg.env.num_nav_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_orig_nav_actions = torch.zeros(self.num_envs, self.cfg.env.num_nav_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_actions = torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device, requires_grad=False)

        self.nav_actions_buffer = torch.zeros(self.num_envs, self.cfg.env.history_len, self.cfg.env.num_nav_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.nav_commands_buffer = torch.zeros(self.num_envs, self.cfg.env.nav_history_len*2, self.cfg.commands.num_nav_commands, dtype=torch.float, device=self.device, requires_grad=False)
        self.delay_nav_commands_hist_buffer = torch.zeros(self.num_envs, self.cfg.env.nav_history_len, self.cfg.commands.num_nav_commands, dtype=torch.float, device=self.device, requires_grad=False)
        self.delay_nav_commands = self.nav_commands_buffer[:, -5, :]
        self.nav_clip_min = torch.tensor([self.cfg.commands.ranges.limit_vx[0], self.cfg.commands.ranges.limit_vy[0], self.cfg.commands.ranges.limit_vyaw[0], self.cfg.commands.ranges.limit_pitch[0]], dtype=torch.float, device=self.device, requires_grad=False)
        self.nav_clip_max = torch.tensor([self.cfg.commands.ranges.limit_vx[1], self.cfg.commands.ranges.limit_vy[1], self.cfg.commands.ranges.limit_vyaw[1], self.cfg.commands.ranges.limit_pitch[1]], dtype=torch.float, device=self.device, requires_grad=False)
        self.obs_hist_buffer = torch.zeros(self.num_envs, self.cfg.env.history_len, self.cfg.env.num_props, dtype=torch.float, device=self.device, requires_grad=False)

        self.euler_rpy = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_lin_vel_pred = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.loco_obs_buf = torch.zeros(
                self.num_envs, 47, device=self.device, dtype=torch.float)
        self.loco_obs_hist = torch.zeros(
                self.num_envs, 10, 47, device=self.device, dtype=torch.float)  
        self.rand_delay_time_ms = torch.randint_like(self.episode_length_buf, low=self.cfg.commands.min_delay_time_ms, high=self.cfg.commands.max_delay_time_ms)

        self.nav_cmds_noise_vec = self._get_nav_cmds_noise_vec()
        self.loco_obs_noise_vec = self._get_loco_obs_noise_scale_vec()

        self._load_loco_policy()

    def _load_loco_policy(self):
        """ load loco policy, which is used to compute loco actions from nav actions
            the loco policy is a slr policy, which is trained with proprioception
        """
        self.loco_body = torch.jit.load('controller/pitch/model.jit')
        self.loco_body =  self.loco_body.to(self.device)

    def _compute_actions(self, nav_actions):
        """ nav_actions (loco_commands) -> loco_actions (self.commands)
            a hacky implementation
        """
        self.commands = self._smooth_nav_actions(nav_actions) # vx, vy, vyaw, pitch
        self._compute_loco_observations()
        actor_obs = self.loco_obs_hist.view(self.num_envs, -1)
        loco_actions, self.base_lin_vel_pred = self.loco_body(actor_obs)
        return loco_actions
    
    def _smooth_nav_actions(self, nav_actions):
        """ Smooth the nav actions over time
        """
        # sample alpha from [0, 1], self.alpha is the smoothing factor, shape: (num_envs, 1)
        # in this way, we can reduce sudden changes in joints (alpha=0: no smoothing, but motors will respond immediately in reality)
        self.alpha = torch_rand_float(0.0, 0.4, (self.num_envs, 1), device=self.device)
        self.nav_actions = self.alpha * nav_actions + (1 - self.alpha) * self.nav_actions
        return self.nav_actions

    def _get_loco_obs_noise_scale_vec(self):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.loco_obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        start = 0
        end = start + self.base_ang_vel.shape[1]
        noise_vec[start:end] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        start = end
        end = start + self.projected_gravity.shape[1]
        noise_vec[start:end] = noise_scales.gravity * noise_level * 1.0
        start = end
        end = start + self.commands.shape[1]
        noise_vec[start:end] = 0. # self.commands
        start = end
        end = start + 1
        noise_vec[start:end] = noise_scales.pitch * noise_level * self.obs_scales.pitch
        start = end
        end = start + self.dof_pos.shape[1]
        noise_vec[start:end] =  noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        start = end
        end = start + self.dof_vel.shape[1]
        noise_vec[start:end] =  noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        start = end
        end = start + self.last_dof_actions.shape[1]
        noise_vec[start:end] = 0. # self.last_dof_actions

        return noise_vec

    def _compute_loco_observations(self):
        """ It is only used for computing loco actions, NOT for updating rl agent.
        """
        props = torch.cat((
                self.base_ang_vel * self.obs_scales.ang_vel, # 3
                self.projected_gravity, # 3
                self.commands * self.commands_scale, # 4
                self.euler_rpy[:, 1:2] * self.obs_scales.pitch,  # dim 1
                self.reindex((self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos),
                self.reindex(self.dof_vel * self.obs_scales.dof_vel),
                self.last_dof_actions),dim=-1)
        
        if self.add_noise:
            props += (2 * torch.rand_like(props) - 1) * self.loco_obs_noise_vec
        
        self.loco_obs_hist = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([props] * self.loco_obs_hist.shape[1], dim=1),
            torch.cat([
                self.loco_obs_hist[:, 1:],
                props.unsqueeze(1)
            ], dim=1)
        )  

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
        # reset object shapes
        self._reset_object_shapes(env_ids)
        # resample object pose and related quantities in world frame
        self._resample_commands(env_ids)

        self.env_start_pos[env_ids] = self.root_states[env_ids, :3].clone() # Store initial pos

        self.last_actions[env_ids] = 0.

        self.last_dof_vel[env_ids] = 0.
        self.last_root_vel[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.object_moved[env_ids] = False
        self.out_of_view_timer[env_ids] = 0.
        self.out_of_view_drift[env_ids] = 0.
        self.success_timer[env_ids] = 0.
        # self.is_success_state[env_ids] = False
        self.loco_obs_hist[env_ids, :, :] = 0.
        self.nav_commands_buffer[env_ids, :, :] = 0.
        self.nav_actions_buffer[env_ids, :, :] = 0.
        self.obs_hist_buffer[env_ids, :, :] = 0.
        self.delay_nav_commands_hist_buffer[env_ids, :, :] = 0.

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        # Log success ratio
        self.extras["success"] = self.is_success_state.float().mean()

    def _roll_robots(self):
        """ Random rolls the robots.
        """
        max_vel_roll = self.cfg.domain_rand.max_vel_roll
        self.root_states[:, 10:11] = torch_rand_float(-max_vel_roll, max_vel_roll, (self.num_envs, 1), device=self.device) # ang vel x: simulate roll
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
           
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
        nav_actions = torch.clip(nav_actions, min=-3.0, max=3.0)
        self.nav_actions_before_clip = nav_actions.to(self.device)
        nav_actions = torch.clip(self.nav_actions_before_clip, min=self.nav_clip_min, max=self.nav_clip_max)
        self.orig_nav_actions = nav_actions.clone()

        actions = self._compute_actions(nav_actions)
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
        roll, pitch, yaw = get_euler_xyz(self.base_quat)
        self.euler_rpy[:, 0] = wrap_to_pi(roll)
        self.euler_rpy[:, 1] = wrap_to_pi(pitch)
        self.euler_rpy[:, 2] = wrap_to_pi(yaw)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.update_image(env_ids=0) # update env0 image for debug viz, instead of all envs
        self.compute_observations() 

        self.last_actions[:] = self.actions[:]
        self.last_orig_nav_actions[:] = self.orig_nav_actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= self.reach_goal

    def _compute_object_center_map(self):
        """ Compute the object center in base frame, camera frame and image plane
            Check if out of view, and apply out-of-view drift if enabled
        """
        # pos in world -> pos in robot
        pos_diff = self.object_pos - self.root_states[:, 0:3]
        self.P_base = quat_rotate_inverse(self.base_quat, pos_diff)
        # compute goal positions in camera frame and image plane
        self.P_camera, self.P_image = self.camera_sensor.transform(self.P_base)

        # Check if physically out of view
        behind_camera = self.P_camera[:, 2] <= 0 # z <= 0
        out_of_bounds = (self.P_image[:, 0] < 0) | (self.P_image[:, 0] > 1) | (self.P_image[:, 1] < 0) | (self.P_image[:, 1] > 1)
        self.physically_out_of_view = behind_camera | out_of_bounds
        
        # Update timer
        self.out_of_view_timer[self.physically_out_of_view] += self.dt
        self.out_of_view_timer[~self.physically_out_of_view] = 0.0
        
        if self.enable_out_of_view_drift:
            # Reset drift for those in view
            self.out_of_view_drift[~self.physically_out_of_view] = 0.0
            
            # Add drift for those out of view
            # drift step: N(0, scale)
            drift_step = torch.randn_like(self.out_of_view_drift[self.physically_out_of_view]) * self.cfg.camera_sensor.drift_scale
            self.out_of_view_drift[self.physically_out_of_view] += drift_step
            
            # Apply drift to P_image
            self.P_image += self.out_of_view_drift
            
        # Determine if we should mask it (return -1)
        threshold = self.cfg.camera_sensor.max_out_of_view_duration if self.enable_out_of_view_drift else 0.0
        should_invalid_mask = behind_camera | (self.out_of_view_timer > threshold)
                
        # Apply mask
        self.P_image[should_invalid_mask] = -1
        
        self.depth_in_view = torch.where(
            should_invalid_mask,
            torch.ones_like(self.P_camera[:, -1]) * -1,  # set to -1 if out of view
            self.P_camera[:, -1].clip(min=0.1)
        )
    
    def _compute_camera_pose_in_world(self):
        """ compute camera pose in world frame
        """
        # compute camera pose in world frame
        # self.camera_position.shape: torch.Size([3]), should be (num_envs, 3)
        # TODO: fix, currently only works for fixed camera position
        self.camera_position_expanded = self.camera_position_scalar.unsqueeze(0).expand(self.num_envs, -1)
        self.camera_world = quat_apply(self.base_quat, self.camera_position_expanded) + self.root_states[:, 0:3]
        R_base_to_world = quat_to_rot_matrix(self.base_quat) # [N, 3, 3]
        R_base_to_cam = self.camera_sensor.R # [N, 3, 3]
        self.R_world_to_cam = torch.bmm(R_base_to_cam, R_base_to_world.transpose(1, 2))
        self.camera_transform_world = {
            'R': self.R_world_to_cam,
            'T': self.camera_world
        }
    
    def resample_object_pose_on_the_way(self):
        """ Resample object pose once per episode when during navigation
        """
        # Avoid resampling immediately after reset (e.g. first 100 steps)
        valid_time = self.episode_length_buf > 100
        to_move =  (~self.object_moved) & valid_time
        env_ids_resample = to_move.nonzero(as_tuple=False).flatten()
        
        if len(env_ids_resample) > 0:
            # Calculate success ratio
            success_ratio = self.is_success_state.float().mean().item()
            
            # Check config for active success feeding
            use_success_feeding = False
            if success_ratio < 0.1:
                use_success_feeding = True
        
            if use_success_feeding:
                # Only feed a subset of environments to encourage exploration
                # Probability of feeding: 30%
                feeding_prob = 0.3
                should_feed = torch.rand(len(env_ids_resample), device=self.device) < feeding_prob
                ids_to_feed = env_ids_resample[should_feed]
                
                if len(ids_to_feed) > 0:
                    self._resample_object_pose_near_success(ids_to_feed)
            else:
                self._resample_commands(env_ids_resample)
                self.env_start_pos[env_ids_resample] = self.root_states[env_ids_resample, :3].clone()
            
            self.object_moved[env_ids_resample] = True

    def _resample_object_pose_near_success(self, env_ids):
        """
        Resample object pose such that the robot is on the path between Hint and Optimal.
        This "feeds" the robot a near-success state.
        """
        if len(env_ids) == 0:
            return

        # 1. Get Robot Position (Gripper)
        robot_pos = self.gripper_world[env_ids] # [N, 3]
        
        # 2. Determine Direction
        # Use robot's current heading (Yaw) to define the path direction
        q_robot = self.root_states[env_ids, 3:7]
        _, _, yaw = get_euler_xyz(q_robot)
        
        # Direction vector (XY plane)
        dir_x = torch.cos(yaw)
        dir_y = torch.sin(yaw)
        dir_z = torch.zeros_like(dir_x)
        path_dir = torch.stack([dir_x, dir_y, dir_z], dim=-1) # [N, 3]
        
        # 3. Handle Place Tasks
        is_place_local = self.is_place[env_ids]
        if is_place_local.any():
            self._resample_near_success_place_object(env_ids, is_place_local, robot_pos, path_dir, yaw)
            
        # 4. Handle Pick Tasks
        is_pick_local = self.is_pick[env_ids]
        if is_pick_local.any():
            # Split into Long and Short
            long_mask = self.obj_is_too_long[env_ids] & is_pick_local
            short_mask = (~self.obj_is_too_long[env_ids]) & is_pick_local
            
            if long_mask.any():
                self._resample_near_success_long_object(env_ids, long_mask, robot_pos, path_dir, yaw)
            
            if short_mask.any():
                self._resample_near_success_short_object(env_ids, short_mask, robot_pos, path_dir, yaw)

        # Update derived quantities
        self._compute_object_axis_end_points_in_world()

    def _resample_near_success_place_object(self, env_ids, mask, robot_pos, path_dir, yaw):
        """ Helper for place object resampling (Large Box) """
        ids_place = env_ids[mask]
        n_place = len(ids_place)
        
        # For Place task, we want the box to be in front of the robot.
        # Similar to short object, but we need to account for box size.
        
        # Sample dist_fwd (distance to goal center): 
        # Box radius approx 0.1 to 0.3.
        # Robot reach approx 0.5.
        # We want box surface to be at reach distance.
        # Center dist = Surface dist + Radius
        
        # Radius estimate (max dim / 2)
        radius = torch.max(self.object_dims[ids_place], dim=1)[0].unsqueeze(-1) / 2.0
        
        # Surface distance: [0.35, 0.65]
        dist_surface = torch_rand_float(0.35, 0.65, (n_place, 1), device=self.device)
        
        dist_fwd = dist_surface + radius
        
        # Sample dist_lat (lateral offset): [-0.1, 0.1]
        dist_lat = torch_rand_float(-0.1, 0.1, (n_place, 1), device=self.device)
            
        dir_place = path_dir[mask]
        
        # Compute Lateral Direction: (-sin, cos, 0)
        dir_lat = torch.stack([-dir_place[:, 1], dir_place[:, 0], torch.zeros_like(dir_place[:, 0])], dim=-1)
        
        robot_pos_place = robot_pos[mask]
        
        # New Object Pos
        obj_pos_xy = robot_pos_place[:, :2] + \
                     dir_place[:, :2] * dist_fwd + \
                     dir_lat[:, :2] * dist_lat
                     
        self.object_pos[ids_place, 0] = obj_pos_xy[:, 0]
        self.object_pos[ids_place, 1] = obj_pos_xy[:, 1]
        
        self.env_start_pos[ids_place] = self.root_states[ids_place, :3].clone()
        
        # Reset Orientation: Align Global X with Path Dir (Upright)
        # Place boxes are forced horizontal, so we just align yaw.
        yaw_place = yaw[mask]
        sy = torch.sin(yaw_place * 0.5)
        cy = torch.cos(yaw_place * 0.5)
        q_yaw = torch.stack([torch.zeros_like(sy), torch.zeros_like(sy), sy, cy], dim=-1)
        
        self.object_quat[ids_place] = q_yaw

    def _resample_near_success_long_object(self, env_ids, mask, robot_pos, path_dir, yaw):
        """ Helper for long object resampling """
        ids_long = env_ids[mask]
        n_long = len(ids_long)
        
        # Use fixed offset for "easy success"
        # offset = self.cfg.env.grasp_offset_long
        
        # Randomize forward distance: [0.35, 0.65]
        dist_fwd = torch_rand_float(0.35, 0.65, (n_long, 1), device=self.device)
        
        # Randomize lateral offset: [-0.1, 0.1]
        dist_lat = torch_rand_float(-0.1, 0.1, (n_long, 1), device=self.device)
        
        # Object Length
        L = self.max_dim_vals[ids_long].unsqueeze(-1)
        
        # Center Position: Center = Robot + Dir_Fwd * (L/2 + dist_fwd) + Dir_Lat * dist_lat
        dir_long = path_dir[mask] # [N, 3] (cos, sin, 0)
        
        # Compute Lateral Direction: (-sin, cos, 0)
        dir_lat = torch.stack([-dir_long[:, 1], dir_long[:, 0], torch.zeros_like(dir_long[:, 0])], dim=-1)
        
        robot_pos_long = robot_pos[mask]
        
        center_xy = robot_pos_long[:, :2] + \
                    dir_long[:, :2] * (L / 2.0 + dist_fwd) + \
                    dir_lat[:, :2] * dist_lat
        
        self.object_pos[ids_long, 0] = center_xy[:, 0]
        self.object_pos[ids_long, 1] = center_xy[:, 1]
        
        # Orientation: Align Long Axis with Path Dir
        # 1. Base rotation: Align Global X with Path Dir
        yaw_long = yaw[mask]
        sy = torch.sin(yaw_long * 0.5)
        cy = torch.cos(yaw_long * 0.5)
        q_yaw = torch.stack([torch.zeros_like(sy), torch.zeros_like(sy), sy, cy], dim=-1)
        
        # 2. Correction rotation: Align Object Long Axis with Global X
        # Find which axis is the long axis
        long_axis_idx = self.max_dim_inds[ids_long] # [N]
        
        # Prepare correction quaternions
        # Case 0: X is long (Identity) -> [0, 0, 0, 1]
        q_corr_0 = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device, dtype=torch.float).repeat(n_long, 1)
        
        # Case 1: Y is long (Rotate -90 deg around Z) -> [0, 0, -0.707, 0.707]
        val_1 = np.sin(-np.pi / 4)
        c_val_1 = np.cos(-np.pi / 4)
        q_corr_1 = torch.tensor([0.0, 0.0, val_1, c_val_1], device=self.device, dtype=torch.float).repeat(n_long, 1)
        
        # Case 2: Z is long (Rotate 90 deg around Y) -> [0, 0.707, 0, 0.707]
        val_2 = np.sin(np.pi / 4)
        c_val_2 = np.cos(np.pi / 4)
        q_corr_2 = torch.tensor([0.0, val_2, 0.0, c_val_2], device=self.device, dtype=torch.float).repeat(n_long, 1)
        
        # Select based on index
        q_corr = torch.where(
            (long_axis_idx == 1).unsqueeze(-1),
            q_corr_1,
            torch.where(
                (long_axis_idx == 2).unsqueeze(-1),
                q_corr_2,
                q_corr_0
            )
        )
        
        # Combine: q_final = q_yaw * q_corr
        self.object_quat[ids_long] = quat_mul(q_yaw, q_corr)

    def _resample_near_success_short_object(self, env_ids, mask, robot_pos, path_dir, yaw):
        """ Helper for short object resampling """
        ids_short = env_ids[mask]
        n_short = len(ids_short)
        
        # Sample dist_fwd (distance to goal): [0.35, 0.65]
        dist_fwd = torch_rand_float(0.35, 0.65, (n_short, 1), device=self.device)
        
        # Sample dist_lat (lateral offset): [-0.1, 0.1]
        dist_lat = torch_rand_float(-0.1, 0.1, (n_short, 1), device=self.device)
            
        dir_short = path_dir[mask]
        
        # Compute Lateral Direction: (-sin, cos, 0)
        dir_lat = torch.stack([-dir_short[:, 1], dir_short[:, 0], torch.zeros_like(dir_short[:, 0])], dim=-1)
        
        robot_pos_short = robot_pos[mask]
        
        # New Object Pos
        obj_pos_xy = robot_pos_short[:, :2] + \
                     dir_short[:, :2] * dist_fwd + \
                     dir_lat[:, :2] * dist_lat
                     
        self.object_pos[ids_short, 0] = obj_pos_xy[:, 0]
        self.object_pos[ids_short, 1] = obj_pos_xy[:, 1]
        
        self.env_start_pos[ids_short] = self.root_states[ids_short, :3].clone()
        
        # Reset Orientation: Align Global X with Path Dir (Upright)
        # This ensures that even if the object tumbled, it is reset to a clean state.
        yaw_short = yaw[mask]
        sy = torch.sin(yaw_short * 0.5)
        cy = torch.cos(yaw_short * 0.5)
        q_yaw = torch.stack([torch.zeros_like(sy), torch.zeros_like(sy), sy, cy], dim=-1)
        
        self.object_quat[ids_short] = q_yaw


    def _compute_optimal_grasp_pose(self):
        """
        Compute the optimal grasp pose and hint pose based on object shape.
        Refactored to decouple Long Axis and Short Axis logic.
        """
        # 1. Compute for Long Axis Objects
        opt_pos_long, opt_quat_long, opt_dir_long, hint_pos_long, hint_quat_long = self._compute_long_object_logic()
        
        # 2. Compute for Short Axis Objects
        opt_pos_short, opt_quat_short, opt_dir_short, hint_pos_short, hint_quat_short = self._compute_short_object_logic()
        
        # 3. Compute for Place Task (Large Box)
        opt_pos_place, opt_quat_place, opt_dir_place, hint_pos_place, hint_quat_place = self._compute_place_task_logic()
        
        # 4. Select based on shape type (Long vs Short) and Task (Pick vs Place)
        # Use short logic if it is not long (small object)
        # Note: Spheres are assumed to be covered by ~self.obj_is_too_long
        use_short_logic = (~self.obj_is_too_long).unsqueeze(-1)
        
        # Pick Task Selection
        opt_pos_pick = torch.where(use_short_logic, opt_pos_short, opt_pos_long)
        opt_quat_pick = torch.where(use_short_logic, opt_quat_short, opt_quat_long)
        opt_dir_pick = torch.where(use_short_logic, opt_dir_short, opt_dir_long)
        hint_pos_pick = torch.where(use_short_logic, hint_pos_short, hint_pos_long)
        hint_quat_pick = torch.where(use_short_logic, hint_quat_short, hint_quat_long)
        
        
        # Final Selection based on Task Flag
        is_place_expanded = self.is_place.unsqueeze(-1)
        
        self.optimal_grasp_pos = torch.where(is_place_expanded, opt_pos_place, opt_pos_pick)
        self.optimal_grasp_pos[:, 2] = torch.where(self.is_place, self.optimal_grasp_pos[:, 2], torch.tensor(0.30, device=self.device)) # Keep 0.30 for Pick, use calculated for Place
        
        self.optimal_grasp_quat = torch.where(is_place_expanded, opt_quat_place, opt_quat_pick)
        self.optimal_approach_dir = torch.where(is_place_expanded, opt_dir_place, opt_dir_pick)
        
        self.hint_pos = torch.where(is_place_expanded, hint_pos_place, hint_pos_pick)
        self.hint_pos[:, 2] = torch.where(self.is_place, self.hint_pos[:, 2], torch.tensor(0.30, device=self.device))
        
        self.hint_quat = torch.where(is_place_expanded, hint_quat_place, hint_quat_pick)
        
    def _compute_place_task_logic(self):
        """
        Compute Optimal Pose and Hint Pose for Place Task (Large Box).
        Target: Slightly above the top edge of the nearest face.
        """
        clearance = self.cfg.target.shape.place_clearance
        hint_dist = 0.4 # Fixed hint distance
        
        # 1. Identify Nearest Face
        # Vector from Box Center to Robot
        vec_box_to_robot = self.root_states[:, :3] - self.object_pos
        
        # Transform to Box Local Frame
        # R_world_to_box = R_box_to_world^T
        # R_box_to_world is from object_quat
        vec_local = quat_rotate_inverse(self.object_quat, vec_box_to_robot) # [N, 3]
        
        # Box Dimensions (Full Extents)
        dims = self.object_dims # [N, 3] (x, y, z)
        
        # Find which axis (x or y) is dominant in local frame (ignoring z for face selection)
        # We assume the robot is on the ground, so we approach from side faces (X or Y)
        abs_vec_local = torch.abs(vec_local)
        is_x_face = abs_vec_local[:, 0] > abs_vec_local[:, 1]
        
        # Determine sign
        sign_x = torch.sign(vec_local[:, 0])
        sign_y = torch.sign(vec_local[:, 1])
        
        # Nearest Face Center in Local Frame
        # If X face: [sign_x * dims[0]/2, 0, 0]
        # If Y face: [0, sign_y * dims[1]/2, 0]
        
        # We want the TOP EDGE of this face.
        # Z coordinate is always +dims[2]/2
        
        # Target Point in Local Frame:
        # X Face: [sign_x * dims[0]/2, 0, dims[2]/2 + clearance]
        # Y Face: [0, sign_y * dims[1]/2, dims[2]/2 + clearance]
        # Note: We center on the edge horizontally? Yes, "nearest face".
        
        target_local = torch.zeros_like(vec_local)
        target_local[:, 2] = dims[:, 2] / 2.0 + clearance
        
        mask_x = is_x_face.unsqueeze(-1)
        
        # X Face
        target_local[:, 0] = torch.where(is_x_face, sign_x * dims[:, 0] / 2.0, target_local[:, 0])
        
        # Y Face
        target_local[:, 1] = torch.where(~is_x_face, sign_y * dims[:, 1] / 2.0, target_local[:, 1])
        
        # 2. Transform to World Frame
        target_world = self.object_pos + quat_apply(self.object_quat, target_local)
        
        # 3. Optimal Direction (Facing the box center from the target)
        # Vector from Target to Box Center (approx)
        # Actually, we want to face "down and in".
        # Let's just face towards the box center in XY, and look down.
        vec_target_to_center = self.object_pos - target_world
        vec_target_to_center[:, 2] = 0 # XY only for yaw
        
        # Safety for zero vector (if target is exactly above center)
        norm_vec = torch.norm(vec_target_to_center, dim=-1, keepdim=True)
        mask_zero = norm_vec.squeeze() < 1e-6
        
        opt_dir = vec_target_to_center / (norm_vec + 1e-6)
        
        if mask_zero.any():
            # Default to forward X if undefined
            opt_dir[mask_zero] = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        
        # 4. Dynamic Pitch Calculation
        # Source: Robot Gripper (approx) or Base + Offset
        # Let's use current gripper position to calculate required pitch?
        # Or a nominal "ready" position?
        # User said: "pitch_target needs to be calculated based on: robot gripper position, target position..."
        
        # Vector from Gripper to Target
        vec_grip_to_target = target_world - self.gripper_world
        d_xy = torch.norm(vec_grip_to_target[:, :2], dim=-1)
        d_z = vec_grip_to_target[:, 2]
        
        # Pitch = atan2(-dz, dxy)
        # If target is lower (dz < 0), pitch > 0 (look down)
        # If target is higher (dz > 0), pitch < 0 (look up)
        # Note: This calculates the pitch of the vector connecting gripper to target.
        # But we want the gripper ORIENTATION pitch.
        # If we want to GRASP/PLACE at that target, we usually want to align with that vector?
        # Or do we want to look down into the box?
        # "gripper needs to be slightly higher than box edge... otherwise can't drop inside"
        # This implies we are dropping FROM above.
        # So the approach should be somewhat downward.
        
        # Let's define pitch based on the trajectory arc?
        # If we are simply calculating the orientation AT the target:
        # For placing, we probably want to point somewhat down (positive pitch).
        # Let's use the vector from Robot Base (shoulder height) to Target.
        # Robot shoulder height ~ 0.4m?
        # If box is 0.6m, target is 0.7m. Robot needs to look UP? Or lift hand up and look down?
        # Usually you lift hand high and look down.
        
        # Let's implement the user's request: "calculated based on gripper pos, target pos..."
        # I will assume we want the gripper to point ALONG the line from current gripper pos to target pos.
        # This makes the robot "point" at the target.
        pitch_dynamic = torch.atan2(-d_z, d_xy)
        
        # Clamp pitch to reasonable values? e.g. [-1.0, 1.0]
        # pitch_dynamic = torch.clamp(pitch_dynamic, -1.0, 1.0)
        
        # 5. Compute Quaternion
        opt_quat = self._compute_quat_from_dir(opt_dir, pitch_dynamic)
        
        # 6. Hint Pose
        # Position: Target + Hint_Dist * (-Opt_Dir) (Back away from box)
        hint_pos = target_world - opt_dir * hint_dist
        hint_quat = opt_quat.clone()
        
        return target_world, opt_quat, opt_dir, hint_pos, hint_quat

    def _compute_long_object_logic(self):
        """
        Compute Optimal Pose and Hint Pose for Long Axis Objects.
        Strategy: Arc approach.
        """
        offset = self.cfg.env.grasp_offset_long
        hint_dist = self.cfg.env.hint_dist_long
        
        # Dynamic Pitch for Pick Task too?
        # User said: "For pick task, target position is kept... pitch_target needs to be calculated..."
        # So we apply dynamic pitch here too.
        
        # 1. Determine Head vs Tail
        # ... (Existing logic to find closest vertex)
        dist_head = torch.norm(self.p_head_world - self.root_states[:, :3], dim=-1)
        dist_tail = torch.norm(self.p_tail_world - self.root_states[:, :3], dim=-1)
        
        head_closer = dist_head < dist_tail
        
        # 2. Define Outward Axis (Along Long Axis, away from center)
        outward_dir = torch.where(
            head_closer.unsqueeze(-1),
            self.axis_long_world,
            -self.axis_long_world
        )
        
        # 3. Optimal Pose
        closest_vertex = torch.where(
            head_closer.unsqueeze(-1),
            self.p_head_world,
            self.p_tail_world
        )
        
        opt_pos = closest_vertex + outward_dir * offset
        opt_dir = -outward_dir # Face towards object
        
        # Dynamic Pitch Calculation
        # Vector from Gripper to Target
        # Note: opt_pos z is usually 0 (ground) or low.
        # Gripper is usually higher.
        # So dz < 0 -> Pitch > 0 (Look down).
        vec_grip_to_target = opt_pos - self.gripper_world
        d_xy = torch.norm(vec_grip_to_target[:, :2], dim=-1)
        d_z = vec_grip_to_target[:, 2]
        pitch_dynamic = torch.atan2(-d_z, d_xy)
        
        # 4. Hint Pose
        hint_pos = closest_vertex + outward_dir * hint_dist
        
        # 5. Compute Quaternions
        opt_quat = self._compute_quat_from_dir(opt_dir, pitch_dynamic)
        hint_quat = opt_quat.clone()
        
        return opt_pos, opt_quat, opt_dir, hint_pos, hint_quat

    def _compute_short_object_logic(self):
        """
        Compute Optimal Pose and Hint Pose for Short Axis Objects.
        Strategy: Straight line approach from start point.
        """
        offset = self.cfg.env.grasp_offset_short
        hint_dist_from_start = self.cfg.env.hint_dist_short
        
        # 1. Define Approach Axis (Start -> Object)
        vec_start_to_obj = self.object_pos - self.env_start_pos
        vec_start_to_obj[:, 2] = 0.0 # XY plane only
        dist_start_to_obj = torch.norm(vec_start_to_obj, dim=-1, keepdim=True)
        axis_inward = vec_start_to_obj / (dist_start_to_obj + 1e-6)
        
        # 2. Optimal Pose
        radius = torch.max(self.tracker.object_dims, dim=1)[0].unsqueeze(-1) / 2.0
        opt_pos = self.object_pos - axis_inward * (radius + offset)
        opt_dir = axis_inward # Face towards object
        
        # Dynamic Pitch Calculation
        vec_grip_to_target = opt_pos - self.gripper_world
        d_xy = torch.norm(vec_grip_to_target[:, :2], dim=-1)
        d_z = vec_grip_to_target[:, 2]
        pitch_dynamic = torch.atan2(-d_z, d_xy)
        
        # 3. Hint Pose
        hint_pos = self.env_start_pos + axis_inward * hint_dist_from_start
        
        # 4. Compute Quaternions
        opt_quat = self._compute_quat_from_dir(opt_dir, pitch_dynamic)
        hint_quat = opt_quat.clone()
        
        return opt_pos, opt_quat, opt_dir, hint_pos, hint_quat

    def _compute_short_object_logic(self):
        """
        Compute Optimal Pose and Hint Pose for Short Axis Objects.
        Strategy: Straight line approach from start point.
        """
        offset = self.cfg.env.grasp_offset_short
        hint_dist_from_start = self.cfg.env.hint_dist_short
        
        # 1. Define Approach Axis (Start -> Object)
        # This is fixed based on initial position to avoid orbiting
        vec_start_to_obj = self.object_pos - self.env_start_pos
        vec_start_to_obj[:, 2] = 0.0 # XY plane only
        dist_start_to_obj = torch.norm(vec_start_to_obj, dim=-1, keepdim=True)
        axis_inward = vec_start_to_obj / (dist_start_to_obj + 1e-6)
        
        # 2. Optimal Pose
        # Position: Object_Surface + Offset * Outward
        # Outward = -Axis_Inward
        # Object_Surface = Object_Center - Radius * Outward? No.
        # Object_Surface = Object_Center - Radius * Axis_Inward (Towards Start)
        
        # Radius estimate (max dim / 2)
        radius = torch.max(self.tracker.object_dims, dim=1)[0].unsqueeze(-1) / 2.0
        
        # Target is at: Center - (Radius + Offset) * Axis_Inward
        # This places it between Start and Object
        opt_pos = self.object_pos - axis_inward * (radius + offset)
        opt_dir = axis_inward # Face towards object
        
        # Dynamic Pitch Calculation
        vec_grip_to_target = opt_pos - self.gripper_world
        d_xy = torch.norm(vec_grip_to_target[:, :2], dim=-1)
        d_z = vec_grip_to_target[:, 2]
        pitch_dynamic = torch.atan2(-d_z, d_xy)
        
        # 3. Hint Pose
        # Position: Start + Hint_Dist * Axis_Inward
        # This places it on the line, near the start
        hint_pos = self.env_start_pos + axis_inward * hint_dist_from_start
        
        # 4. Compute Quaternions
        opt_quat = self._compute_quat_from_dir(opt_dir, pitch_dynamic)
        hint_quat = opt_quat.clone()
        
        return opt_pos, opt_quat, opt_dir, hint_pos, hint_quat

    def _compute_quat_from_dir(self, direction, pitch):
        """ Helper to compute quaternion from direction vector and pitch """
        # Yaw: atan2(dir.y, dir.x)
        yaw = torch.atan2(direction[:, 1], direction[:, 0])
        
        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)
        
        # Ensure pitch is a tensor
        if isinstance(pitch, float):
            pitch = torch.tensor(pitch, device=self.device)
            
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)
        
        # q = [-sp*sy, sp*cy, cp*sy, cp*cy] (Roll=0)
        q_x = -sp * sy
        q_y = sp * cy
        q_z = cp * sy
        q_w = cp * cy
        
        return torch.stack([q_x, q_y, q_z, q_w], dim=-1)

    def _check_success_condition(self):
        """ Check if the robot has reached the success state (close, aligned, immobile)
            New Logic:
            1. Distance to Optimal Grasp Pose < Threshold
            2. Orientation Alignment < Threshold
            3. Velocity < Threshold (Immobile)
        """
        # 1. Distance Check (Close to Optimal Grasp Pose)
        self.dist_to_opt = torch.norm(self.gripper_world[:, :2] - self.optimal_grasp_pos[:, :2], dim=1)
        is_at_target = self.dist_to_opt < 0.05 # 5cm tolerance

        # 2. Orientation Alignment
        q_robot = self.root_states[:, 3:7]
        q_target = self.optimal_grasp_quat
        _, _, yaw_r = get_euler_xyz(q_robot)
        _, _, yaw_t = get_euler_xyz(q_target)
        self.yaw_err = torch.abs(wrap_to_pi(yaw_r - yaw_t))
        is_aligned = self.yaw_err < 0.1 # ~5.7 degrees tolerance

        # 3. Immobile Check
        lin_vel_norm = torch.norm(self.base_lin_vel, dim=1)
        ang_vel_norm = torch.norm(self.base_ang_vel, dim=1)
        is_immobile = (lin_vel_norm < 0.1) & (ang_vel_norm < 0.1) # small velocity tolerance

        return is_at_target & is_aligned & is_immobile

    def _post_physics_step_callback(self):
        """ origin: resample_cmds[env_ids] at resampling_time
            new: update nav_commands every step
        """
        if self.cfg.commands.resample_on_the_way:
            self.resample_object_pose_on_the_way()

        # update distance and reach_goal
        self.distance = torch.norm(self.root_states[:, :2] - self.object_pos[:, :2], dim=1)
        # Robot forward vector in world frame
        self.forward_world = quat_apply(self.base_quat, self.forward_vec)

        # Compute Gripper Position in World Frame
        self.gripper_world = self.root_states[:, :3] + quat_apply(self.base_quat, self.gripper_offset)

        # get self.camera_transform_world('R': self.R_world_to_cam, 'T': self.camera_world)
        self._compute_camera_pose_in_world()

        # map object center to image plane and get depth_in_view
        self._compute_object_center_map() # useless now

        # # get optimal grasp pose
        self._compute_optimal_grasp_pose() # nontrivial

        # get self.sigma_points_3d, self.sigma_points_base, self.sigma_points_2d, self.is_valid for perception in obs
        self._get_nav_commands() # nontrivial

        self.nav_actions_buffer = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([self.nav_actions] * self.nav_actions_buffer.shape[1], dim=1),
            torch.cat([
                self.nav_actions_buffer[:, 1:],
                self.nav_actions.unsqueeze(1)
            ], dim=1)
        ) 

        self.timer = (self.episode_length_buf / self.max_episode_length).unsqueeze(-1)

        # --- New Termination Logic (Success Check) ---
        self.is_success_state = self._check_success_condition()

        # Update timer
        self.success_timer[self.is_success_state] += self.dt
        self.success_timer[~self.is_success_state] = 0.0
        
        # Terminate only after holding for 2 seconds
        self.reach_goal = self.success_timer > 2.0
    
    def _reset_object_shapes(self, env_ids):
        if len(env_ids) == 0:
            return

        target_cfg = self.cfg.target
        
        # --- Task Sampling ---
        # 0: Pick, 1: Place
        # 1/3 Place, 2/3 Pick
        probs = torch.rand(len(env_ids), device=self.device)
        is_place_local = probs < target_cfg.init.place_prob # shape: [len(env_ids)]
        
        self.is_place[env_ids] = is_place_local
        self.is_pick[env_ids] = ~is_place_local
        
        self.task_flags[env_ids] = is_place_local.float().unsqueeze(-1)

        # --- Shape Type Selection ---
        # Default: Random mixed
        type_indices = torch.randint(0, len(self.shape_types_list), (len(env_ids),), device=self.device)
        
        # Override for Place Task: Must be Cuboid (Box)
        if "cuboid" in self.shape_types_list:
            cub_idx = self.shape_types_list.index("cuboid")
            type_indices[is_place_local] = cub_idx
            
        self.env_shape_type_indices[env_ids] = type_indices
        
        # 2. Iterate over types
        for i, type_name in enumerate(self.shape_types_list):
            subset_indices = self.env_shape_type_indices[env_ids]
            mask = (subset_indices == i)
            
            if not mask.any():
                continue
            
            # Get subset for this type
            current_ids = env_ids[mask]
            current_is_place = is_place_local[mask]
            count = len(current_ids)
            
            # Generate random params
            if type_name == "cylinder":
                r = torch_rand_float(target_cfg.shape.radius_range[0], target_cfg.shape.radius_range[1], (count, 1), device=self.device)
                h = torch_rand_float(target_cfg.shape.height_range[0], target_cfg.shape.height_range[1], (count, 1), device=self.device)
                params = torch.cat([r, h], dim=1)
            elif type_name == "cuboid":
                # Sample Small (Pick)
                x_s = torch_rand_float(target_cfg.shape.dims_range[0][0], target_cfg.shape.dims_range[0][1], (count, 1), device=self.device)
                y_s = torch_rand_float(target_cfg.shape.dims_range[1][0], target_cfg.shape.dims_range[1][1], (count, 1), device=self.device)
                z_s = torch_rand_float(target_cfg.shape.dims_range[2][0], target_cfg.shape.dims_range[2][1], (count, 1), device=self.device)
                
                # Sample Large (Place)
                x_l = torch_rand_float(target_cfg.shape.box_dims_range[0][0], target_cfg.shape.box_dims_range[0][1], (count, 1), device=self.device)
                y_l = torch_rand_float(target_cfg.shape.box_dims_range[1][0], target_cfg.shape.box_dims_range[1][1], (count, 1), device=self.device)
                z_l = torch_rand_float(target_cfg.shape.box_dims_range[2][0], target_cfg.shape.box_dims_range[2][1], (count, 1), device=self.device)
                
                # Select based on task
                x = torch.where(current_is_place.unsqueeze(-1), x_l, x_s)
                y = torch.where(current_is_place.unsqueeze(-1), y_l, y_s)
                z = torch.where(current_is_place.unsqueeze(-1), z_l, z_s)
                
                params = torch.cat([x, y, z], dim=1)
            elif type_name == "sphere":
                r = torch_rand_float(target_cfg.shape.radius_range[0], target_cfg.shape.radius_range[1], (count, 1), device=self.device)
                params = r
            
            # Compute dims
            dims = PCATargetTracker.compute_object_dims(type_name, params, self.device)
            
            # Sample points
            shape = self.shapes[type_name]
            pts, nrms = shape.sample_surface(target_cfg.perception.num_sample_points, count, params)
            
            # Update buffers
            self.object_dims[current_ids] = dims
            self.local_points[current_ids] = pts
            self.local_normals[current_ids] = nrms
            
        # Update tracker if it exists
        if hasattr(self, 'tracker'):
            self.tracker.object_dims[env_ids] = self.object_dims[env_ids]
            self.tracker.local_points[env_ids] = self.local_points[env_ids]
            self.tracker.local_normals[env_ids] = self.local_normals[env_ids]

        self.max_dim_vals[env_ids], self.max_dim_inds[env_ids] = torch.max(self.object_dims[env_ids], dim=1) # [N], [N]
        
        # Determine Vertical/Horizontal (50% chance)
        random_vertical = (torch.rand(len(env_ids), device=self.device) < 0.5)
        # Force Place Task to be Horizontal (Standard orientation)
        # This ensures Local Z is World Z, simplifying target calculation
        random_vertical[is_place_local] = False
        self.obj_is_vertical[env_ids] = random_vertical
        
        is_horizontal = ~self.obj_is_vertical[env_ids]
        is_large = self.max_dim_vals[env_ids] > self.gripper_width
        
        # Check if Sphere
        is_sphere = torch.zeros_like(is_horizontal, dtype=torch.bool)
        if "sphere" in self.shape_types_list:
            sphere_idx = self.shape_types_list.index("sphere")
            is_sphere = (self.env_shape_type_indices[env_ids] == sphere_idx)
            
        # Too Long = Horizontal AND Large AND Not Sphere
        self.obj_is_too_long[env_ids] = is_horizontal & is_large & (~is_sphere)
        
        # Compute Z Offset
        z_off = torch.zeros_like(self.object_dims[env_ids, 2])
        
        if "cylinder" in self.shape_types_list:
            cyl_idx = self.shape_types_list.index("cylinder")
            is_cyl = (self.env_shape_type_indices[env_ids] == cyl_idx)
            
            # Vertical: h/2 (dims[2]/2)
            # Horizontal: d/2 (dims[0]/2)
            mask_v = is_cyl & self.obj_is_vertical[env_ids]
            mask_h = is_cyl & (~self.obj_is_vertical[env_ids])
            
            if mask_v.any():
                z_off[mask_v] = self.object_dims[env_ids][mask_v, 2] / 2.0
            if mask_h.any():
                z_off[mask_h] = self.object_dims[env_ids][mask_h, 0] / 2.0
            
        if "cuboid" in self.shape_types_list:
            cub_idx = self.shape_types_list.index("cuboid")
            is_cub = (self.env_shape_type_indices[env_ids] == cub_idx)
            
            # Vertical: y/2 (dims[1]/2)
            # Horizontal: z/2 (dims[2]/2)
            mask_v = is_cub & self.obj_is_vertical[env_ids]
            mask_h = is_cub & (~self.obj_is_vertical[env_ids])
            
            if mask_v.any():
                z_off[mask_v] = self.object_dims[env_ids][mask_v, 1] / 2.0
            if mask_h.any():
                z_off[mask_h] = self.object_dims[env_ids][mask_h, 2] / 2.0
            
        if "sphere" in self.shape_types_list:
            sph_idx = self.shape_types_list.index("sphere")
            is_sph = (self.env_shape_type_indices[env_ids] == sph_idx)
            if is_sph.any():
                z_off[is_sph] = self.object_dims[env_ids][is_sph, 2] / 2.0
            
        self.object_z_offset[env_ids] = z_off

    def _resample_object_positions(self, env_ids):
        """ set the goal position in the visualable zone of the camera
        """
        # 1. Sample u, v in Image Plane (Normalized [0, 1])
        u = torch.rand((len(env_ids), 1), device=self.device)
        v = torch.rand((len(env_ids), 1), device=self.device)
        
        # Use cached tensors
        fx = self.cam_fx[env_ids]
        fy = self.cam_fy[env_ids]
        cx = self.cam_cx[env_ids]
        cy = self.cam_cy[env_ids]
        img_w = self.cam_img_w[env_ids]
        img_h = self.cam_img_h[env_ids]
        
        # 2. Compute ray in Camera Frame (Z-forward)
        # P_cam = [x, y, z] = z * [(u - cx)/fx, (v - cy)/fy, 1]
        # Let dir_cam = [(u - cx)/fx, (v - cy)/fy, 1]
        dir_cam_x = (u * img_w - cx) / fx
        dir_cam_y = (v * img_h - cy) / fy
        dir_cam_z = self.dir_cam_z[env_ids]
        
        dir_cam = torch.cat([dir_cam_x, dir_cam_y, dir_cam_z], dim=-1) # (N, 3)
        
        # 3. Transform ray to World Frame
        # dir_base = R^T * dir_cam
        R = self.camera_sensor.R[env_ids]
        dir_base = torch.bmm(R.transpose(1, 2), dir_cam.unsqueeze(-1)).squeeze(-1)
        
        # dir_world = quat_apply(base_quat, dir_base)
        base_quat = self.base_quat[env_ids]
        dir_world = quat_apply(base_quat, dir_base)
        
        # 4. Camera World Position
        # pos_cam_base = T
        T = self.camera_sensor.T[env_ids]
        base_pos = self.root_states[env_ids, :3]
        pos_cam_world = quat_apply(base_quat, T) + base_pos
        
        # 5. Compute valid depth range [t_min, t_max] for z_cam (which is t)
        # We want P_world.z > height_min
        # P_world.z = pos_cam_world.z + t * dir_world.z
        
        height_min = 0.00 # 0cm above ground
        
        # Default range
        t_min = self.t_min_default[env_ids].clone()
        t_max = self.t_max_default[env_ids].clone()
        
        dz = dir_world[:, 2:3]
        pz = pos_cam_world[:, 2:3]
        
        # Case 1: Ray points down (dz < -1e-4)
        # t < (height_min - pz) / dz
        mask_down = dz < -1e-4
        t_limit_down = (height_min - pz) / dz
        t_max = torch.where(mask_down, torch.min(t_max, t_limit_down), t_max)
        
        # Case 2: Ray points up (dz > 1e-4)
        # t > (height_min - pz) / dz
        mask_up = dz > 1e-4
        t_limit_up = (height_min - pz) / dz
        t_min = torch.where(mask_up, torch.max(t_min, t_limit_up), t_min)
        
        # Ensure t_min <= t_max (prioritize height constraint t_max)
        t_min = torch.min(t_min, t_max)

        # Sample t
        rand_val = torch.rand((len(env_ids), 1), device=self.device)
        t = t_min + rand_val * (t_max - t_min)
        
        # 6. Compute P_world
        P_world = pos_cam_world + t * dir_world
        
        # Use cached z_offset
        P_world[:, 2] = self.object_z_offset[env_ids]

        return P_world
    
    def _resample_object_orientations(self, env_ids):
        """ set the goal orientation
        """
        if len(env_ids) == 0:
            return
        
        # 1. Compute Yaw (Random or Fixed)
        if self.randomize_orientation:
            yaw = torch.rand((len(env_ids), 1), device=self.device) * 2 * np.pi
        else:
            yaw = torch.zeros((len(env_ids), 1), device=self.device)
            
        # 2. Convert to Quaternion (Rotation around Z)
        sy = torch.sin(yaw * 0.5)
        cy = torch.cos(yaw * 0.5)
        q_yaw = torch.cat([torch.zeros_like(sy), torch.zeros_like(sy), sy, cy], dim=-1)
        
        # 3. Apply shape-specific base rotation
        q_final = q_yaw.clone()
        
        # Identify shapes
        is_cylinder = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        is_cuboid = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        
        if "cylinder" in self.shape_types_list:
            cyl_idx = self.shape_types_list.index("cylinder")
            is_cylinder = (self.env_shape_type_indices[env_ids] == cyl_idx)
            
        if "cuboid" in self.shape_types_list:
            cub_idx = self.shape_types_list.index("cuboid")
            is_cuboid = (self.env_shape_type_indices[env_ids] == cub_idx)
            
        is_vertical = self.obj_is_vertical[env_ids]
        is_horizontal = ~is_vertical
        
        # Prepare Rotations
        val = np.sin(np.pi / 4)
        # q_rot_y_90: Rotate 90 deg around Y (Z -> X)
        q_rot_y_90 = torch.tensor([0.0, val, 0.0, val], device=self.device, dtype=torch.float).view(1, 4).repeat(len(env_ids), 1)
        
        # q_rot_x_90: Rotate 90 deg around X (Y -> Z)
        q_rot_x_90 = torch.tensor([val, 0.0, 0.0, val], device=self.device, dtype=torch.float).view(1, 4).repeat(len(env_ids), 1)
        
        # Apply Cylinder Logic
        # Horizontal Cylinder -> Rotate Y 90 (Lay down)
        mask_cyl_horz = is_cylinder & is_horizontal
        if mask_cyl_horz.any():
            q_final[mask_cyl_horz] = quat_mul(q_yaw[mask_cyl_horz], q_rot_y_90[mask_cyl_horz])
            
        # Apply Cuboid Logic
        # Vertical Cuboid -> Rotate X 90 (Stand up, assuming Y is long)
        mask_cub_vert = is_cuboid & is_vertical
        if mask_cub_vert.any():
            q_final[mask_cub_vert] = quat_mul(q_yaw[mask_cub_vert], q_rot_x_90[mask_cub_vert])

        return q_final

    def _compute_object_axis_end_points_in_world(self):
        """ Compute object axes and head/tail points in world frame.
        """
        # Object axes in world frame
        self.x_axis_world = quat_apply(self.object_quat, self.x_axis_local)
        self.y_axis_world = quat_apply(self.object_quat, self.y_axis_local)
        self.z_axis_world = quat_apply(self.object_quat, self.z_axis_local)
        self.all_axes = torch.stack([self.x_axis_world, self.y_axis_world, self.z_axis_world], dim=1)

        batch_indices = torch.arange(self.num_envs, device=self.device)
        self.axis_long_world = self.all_axes[batch_indices, self.max_dim_inds] # [N, 3]
        
        half_length = (self.max_dim_vals / 2.0).unsqueeze(-1)
        self.p_head_world = self.object_pos + self.axis_long_world * half_length
        self.p_tail_world = self.object_pos - self.axis_long_world * half_length

    def _resample_commands(self, env_ids):
        """ Set the object pose in the visualable zone and update related world frame quantities
        """
        if len(env_ids) == 0:
            return
        
        # 1. Resample Object Position in World Frame
        P_world = self._resample_object_positions(env_ids)
        # 2. Resample Object Orientation in World Frame
        q_final = self._resample_object_orientations(env_ids)

        self.object_pos[env_ids] = P_world
        self.object_quat[env_ids] = q_final

        # Get object axis and end points in world frame
        self._compute_object_axis_end_points_in_world()
        
    def _compute_sigma_points_base(self, sigma_points_3d, base_pos, base_quat, is_valid):
        """ Compute perception features (Sigma Points) in Robot Base Frame.
        """
        num_points = sigma_points_3d.shape[1]
        delta = sigma_points_3d - base_pos # [N, M, 3]
        sigma_points_base_flat = quat_rotate_inverse(base_quat.expand(-1, num_points, -1).reshape(-1, 4), delta.reshape(-1, 3))
        current_sigma_base = sigma_points_base_flat.view(self.num_envs, num_points, 3)
        
        # If valid, update buffer. If invalid, set invalid data
        valid_mask = is_valid.view(self.num_envs, 1, 1)
        invalid_sigma = torch.ones_like(current_sigma_base) * -1.0
        self.sigma_points_base = torch.where(valid_mask, current_sigma_base, invalid_sigma)

    def _compute_sigma_points_camera(self):
        """ Compute perception features (Sigma Points) in Camera Frame.
        """
        delta_cam = self.sigma_points_base - self.camera_sensor.T.unsqueeze(1)  # [N, M, 3]
        res = torch.bmm(self.camera_sensor.R, delta_cam.transpose(1, 2)) # [N, 3, M]
        self.sigma_points_camera = res.transpose(1, 2) # [N, M, 3] (x, y, z)
    
    def _compute_sigma_points_image(self):
        """ Compute perception features (Sigma Points) in Image Plane.
        """
        if not hasattr(self, 'sigma_points_camera'):
            raise ValueError("Sigma Points in Camera Frame not computed yet.")
        
        x = self.sigma_points_camera[:, :, 0]
        y = self.sigma_points_camera[:, :, 1]
        z = self.sigma_points_camera[:, :, 2]
        
        # Avoid div by zero
        z_safe = torch.where(z < 1e-5, torch.ones_like(z) * 1e-5, z)
        
        u = (x / z_safe) * self.cam_fx + self.cam_cx
        v = (y / z_safe) * self.cam_fy + self.cam_cy
        
        # Normalize
        u_norm = u / self.cam_img_w
        v_norm = v / self.cam_img_h
        
        self.sigma_points_image = torch.stack([u_norm, v_norm, z], dim=-1) # [N, M, 3]

    def _get_nav_commands(self):
        """ Compute perception features (Sigma Points) and update nav_commands.
        """
        base_pos = self.root_states[:, :3].unsqueeze(1) # [N, 1, 3]
        base_quat = self.root_states[:, 3:7].unsqueeze(1) # [N, 1, 4]

        # 1. Run Perception Tracker to get 3D Sigma Points in World Frame
        # sigma_points_2d: [N, 5, 2] in Image Plane (normalized uv)
        # sigma_points_3d: [N, 7, 3] in World Frame (if 3D)
        # is_valid: [N]
        sigma_points_2d, _, _, _, _, _, sigma_points_3d, is_valid = self.tracker.compute_features(
            object_pos=self.object_pos, # shape: [N, 3]
            object_quat=self.object_quat, # shape: [N, 4]
            camera_params=self.camera_params_dict,
            camera_transform=self.camera_transform_world, # camrea pose in world frame
            use_geometric_weight=self.use_geometric_weight,
            debug_timer=self.debug_timer,
            debug_info=self.debug_info
        )

        if self.cfg.env.use_3d_sigma_points:
            # 3D Mode: Use 3D Sigma Points (7 points) -> Camera Frame
            self._compute_sigma_points_base(sigma_points_3d, base_pos, base_quat, is_valid)
            self._compute_sigma_points_camera()
            self.sigma_points_obs = self.sigma_points_camera # [N, 7, 3]
        else:
            # 2D Mode: Use 2D Sigma Points (5 points) -> Image Plane
            # sigma_points_2d is already in Image Plane (normalized)
            
            # Compute Z depth in Camera Frame
            self._compute_sigma_points_base(sigma_points_3d, base_pos, base_quat, is_valid)
            self._compute_sigma_points_camera()
            z_depth = self.sigma_points_camera[:, :, 2:3] # [N, 5, 1]
            
            # Concatenate (u, v, z)
            sigma_points_obs_unmasked = torch.cat([sigma_points_2d, z_depth], dim=-1) # [N, 5, 3]

            # Handle invalid points
            valid_mask = is_valid.view(self.num_envs, 1, 1)
            invalid_sigma = torch.ones_like(sigma_points_obs_unmasked) * -1.0
            self.sigma_points_obs = torch.where(valid_mask, sigma_points_obs_unmasked, invalid_sigma) # [N, 5, 3]

        # 3. Update nav_commands
        self.nav_commands = self.sigma_points_obs.reshape(self.num_envs, -1)
        
        self.sigma_points_3d = sigma_points_3d
        self.sigma_points_2d = sigma_points_2d
        self.is_valid = is_valid

    def _draw_debug_vis(self, sigma_points_3d=None, sigma_points_2d=None):
        """ Draw Sigma Points in 3D and 2D """
        if sigma_points_3d is None:
            if hasattr(self, 'sigma_points_3d'):
                sigma_points_3d = self.sigma_points_3d
            else:
                return
        
        if sigma_points_2d is None:
            if hasattr(self, 'sigma_points_2d'):
                sigma_points_2d = self.sigma_points_2d
        
        if self.viewer:
            self.gym.clear_lines(self.viewer)

            # Draw env_start_pos
            self.vis_utils.draw_start_pos(
                self.env_start_pos[0],
                env_idx=0
            )
            
            # Draw Optimal Grasp Pose in 3D
            self.vis_utils.draw_optimal_grasp_pose(
                self.optimal_grasp_pos[0],
                self.optimal_grasp_quat[0],
                env_idx=0
            )
            
            # Draw Hint Pose in 3D
            self.vis_utils.draw_hint_pose(
                self.hint_pos[0],
                self.hint_quat[0],
                env_idx=0
            )

            # Draw Sequential Reaching Debug Info
            self.vis_utils.draw_sequential_reaching_debug(
                hint_pos=self.hint_pos[0],
                optimal_pos=self.optimal_grasp_pos[0],
                gripper_pos=self.gripper_world[0],
                env_idx=0
            )

            # Downsample target points for visualization
            points_world_sample = self.vis_utils.draw_target_points(
                local_points=self.tracker.local_points[0],
                object_pos=self.object_pos[0],
                object_quat=self.object_quat[0],
                num_vis_points=self.cfg.camera_sensor.num_vis_points,
                env_idx=0
            )

            # Draw Sigma points in 3D
            if self.cfg.camera_sensor.vis_sigma_3d_in_world:
                self.vis_utils.draw_sigma_points_3d(
                    sigma_points=sigma_points_3d[0], 
                    env_idx=0
                )

            # Draw Sigma points Axes
            if self.cfg.camera_sensor.vis_sigma_axes:
                self.vis_utils.draw_sigma_axes(
                    sigma_points=sigma_points_3d[0], 
                    env_idx=0
                )
            
            # Draw Object Axes
            if self.cfg.camera_sensor.vis_object_axes:
                self.vis_utils.draw_object_axes(
                    object_pos=self.object_pos[0],
                    x_axis=self.x_axis_world[0],
                    y_axis=self.y_axis_world[0],
                    z_axis=self.z_axis_world[0],
                    env_idx=0
                )
            
            # Draw Head and Tail Points
            if self.cfg.camera_sensor.vis_head_tail_points:
                self.vis_utils.draw_head_tail_points(
                    head_pos=self.p_head_world[0],
                    tail_pos=self.p_tail_world[0],
                    env_idx=0
                )
            
            if self.cfg.camera_sensor.vis_gripper_position:
                self.vis_utils.draw_gripper_position(
                    gripper_pos=self.gripper_world[0],
                    env_idx=0
                )
            
            # 3. Project to camera image plane and Draw
            if self.common_step_counter % 10 == 0:
                points_3d_dict = {}
                if self.cfg.camera_sensor.vis_target_points_in_image:
                    points_3d_dict["target"] = points_world_sample
                if self.cfg.camera_sensor.vis_sigma_3d_in_image:
                    points_3d_dict["sigma_3d"] = sigma_points_3d[0]
                
                points_2d_dict = {}
                if self.cfg.camera_sensor.vis_sigma_2d and sigma_points_2d is not None:
                    points_2d_dict = {
                        "sigma_2d": sigma_points_2d[0]
                    }
                
                lines_dict = {}
                if self.cfg.camera_sensor.vis_sigma_y_spread:
                    # Calculate Y-Spread Axis (Visual Foreshortening)
                    points_body = self.sigma_points_base[0] # [5, 3]
                    ys = points_body[:, 1]
                    min_y = torch.min(ys)
                    max_y = torch.max(ys)
                    mean_x = torch.mean(points_body[:, 0])
                    mean_z = torch.mean(points_body[:, 2])
                    
                    p1_body = torch.tensor([mean_x, min_y, mean_z], device=self.device)
                    p2_body = torch.tensor([mean_x, max_y, mean_z], device=self.device)
                                        
                    p1_world = quat_apply(self.base_quat[0], p1_body) + self.root_states[0, :3]
                    p2_world = quat_apply(self.base_quat[0], p2_body) + self.root_states[0, :3]
                    
                    lines_dict["y_spread"] = torch.stack([p1_world, p2_world])

                self.vis_utils.draw_2d_image(
                    points_3d_dict=points_3d_dict, 
                    camera_sensor=self.camera_sensor, 
                    env_idx=0, 
                    filename=f"debug_cam_{self.common_step_counter}.png",
                    save_images=self.save_debug_images,
                    points_2d_dict=points_2d_dict,
                    lines_3d_dict=lines_dict
                )
    
    def _get_nav_cmds_noise_vec(self):
        noise_vec = torch.zeros_like(self.nav_commands[0])
        noise_scales = self.cfg.noise.noise_scales
        noise_vec[0] = noise_scales.nav_cmd_x
        noise_vec[1] = noise_scales.nav_cmd_y
        noise_vec[2] = noise_scales.nav_cmd_z
        return noise_vec

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(self.cfg.env.num_props, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        start = 0
        end = self.base_lin_vel.shape[1]
        noise_vec[start:end] = 0. # self.base_lin_vel_pred (get from loco policy, no noise)

        start = end
        end = self.base_ang_vel.shape[1]
        noise_vec[start:end] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel

        start = end
        end = start + self.projected_gravity.shape[1]
        noise_vec[start:end] = noise_scales.gravity * noise_level * 1.0

        start = end
        end = start + self.euler_rpy.shape[1]
        noise_vec[start:end] = noise_scales.euler_rpy * noise_level * self.obs_scales.euler_rpy

        start = end
        end = start + self.nav_commands.shape[1]
        noise_vec[start:end] = self.nav_cmds_noise_vec * noise_level 

        start = end
        end = start + self.task_flags.shape[1]
        noise_vec[start:end] = 0. # self.task_flags (get from task, no noise)

        start = end
        end = start + self.nav_actions.shape[1]
        noise_vec[start:end] = 0. # self.nav_actions (get from rl policy, no noise)
        return noise_vec

    def _resample_delay_nav_commands(self):
        """ Resample delayed navigation commands
        """
        if self.cfg.commands.enable_delay:
            env_ids = (self.episode_length_buf % (self.rand_delay_time_ms / (self.dt * 1000)).to(dtype=torch.int64) == 0).nonzero(as_tuple=False).flatten()
            if len(env_ids) != 0:
                # cfg.commands.delay_time: fixed delay time, 150 ms
                # self.nav_commands_buffer: step time = 20 ms
                # max_delay_time_ms = 200 ms -> max_delay_step = 10
                # min_delay_time_ms = 100 ms -> min_delay_step = 5
                # delay_time = random(100, 200) ms -> delay_step = random(5, 10)
                # resample nav_commands when delay_time
                max_delay_step = int(self.cfg.commands.max_delay_time_ms // (self.dt * 1000))
                min_delay_step = int(self.cfg.commands.min_delay_time_ms // (self.dt * 1000))
                if max_delay_step > min_delay_step:
                    resample_time_idx = -torch.randint(min_delay_step, max_delay_step, (len(env_ids),), device=self.device) -1 # simulate a small random delay
                elif max_delay_step == min_delay_step:
                    resample_time_idx = -torch.tensor([max_delay_step]*len(env_ids), device=self.device) -1
                else:
                    raise ValueError("max_delay_time_ms must be greater than or equal to min_delay_time_ms")
                self.delay_nav_commands[env_ids] = self.nav_commands_buffer[env_ids, resample_time_idx, :]
                # if 0 in env_ids:
                    #     print(f"buffer: {self.nav_commands_buffer[0]}")
                    #     print(f"resample ({resample_time_idx[0]}): {self.delay_nav_commands[0]}")
        else:
            self.delay_nav_commands = self.nav_commands.clone()

    def compute_observations(self):
        """ Computes observations for updating nav agent
        """
        self._resample_delay_nav_commands()
        obs_buf = torch.cat([
            self.base_lin_vel_pred * self.obs_scales.lin_vel, # 3
            self.base_ang_vel * self.obs_scales.ang_vel, # 3
            self.projected_gravity, # 3
            self.euler_rpy[:, :3] * self.obs_scales.euler_rpy,  # rpy 3
            self.delay_nav_commands, # 3 * self.cfg.env.num_sigma_points = 15 or 21
            self.task_flags, # 1
            self.nav_actions_before_clip # 4
            ], dim=-1)

        # add noise if needed
        if self.add_noise:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec

        self.obs_hist_buffer = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
            torch.cat([
                self.obs_hist_buffer[:, 1:],
                obs_buf.unsqueeze(1)
            ], dim=1)
        )
        
        self.delay_nav_commands_hist_buffer = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([self.delay_nav_commands] * self.cfg.env.nav_history_len, dim=1),
            torch.cat([
                self.delay_nav_commands_hist_buffer[:, 1:],
                self.delay_nav_commands.unsqueeze(1)
            ], dim=1)
        )

        if self.num_obs == self.num_props * self.cfg.env.history_len: # used for RNN
            self.obs_buf = self.obs_hist_buffer.view(self.num_envs, -1)
            self.privileged_obs_buf = torch.cat([
                self.sigma_points_obs.reshape(self.num_envs, -1), self.obs_hist_buffer.view(self.num_envs, -1)], dim=-1)
        else:
            self.obs_buf = torch.cat([
                self.sigma_points_obs.reshape(self.num_envs, -1), self.obs_hist_buffer.view(self.num_envs, -1), self.delay_nav_commands_hist_buffer.view(self.num_envs, -1)
                ], dim=-1)


    # ------------- Cameras -------------
    def attach_camera(self, env_handle, actor_handle):
        if not hasattr(self, 'camera_position') or not hasattr(self, 'camera_angle'):
            self.camera_position_scalar = torch.tensor(self.cfg.camera_sensor.extrinsics.translation, device=self.device, dtype=torch.float)
            self.camera_angle = self.cfg.camera_sensor.extrinsics.angles
            self.enable_camera = self.cfg.camera_sensor.enable_camera
        if not self.enable_camera:
            return
        camera_props = gymapi.CameraProperties()
        camera_props.enable_tensors = True
        camera_props.width = self.cfg.camera_sensor.intrinsics.img_width
        camera_props.height = self.cfg.camera_sensor.intrinsics.img_height        
        camera_props.horizontal_fov = self.cfg.camera_sensor.intrinsics.horizontal_fov
        
        camera_handle = self.gym.create_camera_sensor(
            env_handle, camera_props)
        root_handle = self.gym.get_actor_root_rigid_body_handle(
            env_handle, actor_handle)
        local_transform = gymapi.Transform()
        local_transform.p = gymapi.Vec3(*self.camera_position_scalar)
        local_transform.r = gymapi.Quat.from_euler_zyx(
            np.radians(self.camera_angle[0]), np.radians(self.camera_angle[1]), np.radians(self.camera_angle[2]))

        self.gym.attach_camera_to_body(
            camera_handle, env_handle, root_handle, local_transform, gymapi.FOLLOW_TRANSFORM)

        self.cam_handles.append(camera_handle)
    
    def update_image(self, env_ids=0):
        if not self.enable_camera:
            return
        self.gym.step_graphics(self.sim)  # required to render in headless mode
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        image_ = self.gym.get_camera_image_gpu_tensor(self.sim,
                                                        self.envs[env_ids],
                                                        self.cam_handles[env_ids],
                                                        gymapi.IMAGE_COLOR)
        self.image = gymtorch.wrap_tensor(image_)
        self.gym.end_access_image_tensors(self.sim)

    #### rewards

    def _reward_orientation_y(self):
        return torch.square(self.projected_gravity[:, 1])
    
    def _reward_nav_action_rate(self):
        return torch.square(torch.norm(self.orig_nav_actions - self.last_orig_nav_actions, dim=-1))

    def _reward_nav_action_limit(self):
        return torch.square(torch.norm(self.nav_actions - self.nav_actions_before_clip, dim=-1))

    def _reward_heading_target(self):
        """ Reward for facing the target
        """
        # Calculate the angle of the target in the robot's base frame
        heading_error = torch.abs(torch.atan2(self.P_base[:, 1], self.P_base[:, 0]))
        # forward_w = quat_apply(self.base_quat, self.forward_vec)
        # base_theta = torch.atan2(forward_w[:, 1], forward_w[:, 0])
        # pos_diff_w = (self.object_pos - self.root_states[:, :3])
        # pos_diff_w_xy = pos_diff_w[:, :2] / torch.norm(pos_diff_w[:, :2], dim=1, keepdim=True)
        # goal_theta = torch.atan2(pos_diff_w_xy[:, 1], pos_diff_w_xy[:, 0])
        # theta_error = torch.abs(wrap_to_pi(goal_theta - base_theta))
        return torch.exp(-heading_error / 0.1)
    
    def _reward_view_missing(self):
        """ Reward for missing the goal position in the view
        """
        return self.physically_out_of_view.float()

    def _reward_lin_vel_y(self):
        return torch.abs(self.base_lin_vel[:, 1])
    
    def _reward_tracking_view_center(self):
        """ Reward for align the goal position with the center of the view
        """
        tight_area = self.distance < 0.4
        dy_error = torch.square(self.P_base[:, 1])
        pitch_restricted = torch.logical_and(self.euler_rpy[:, 1] > 0.35, self.object_pos[:, 2] < 0.2)
        return pitch_restricted.float() * tight_area * torch.exp(-dy_error/0.1)
    
    def _reward_tracking_horizontal_distance(self):
        """ Reward for tracking the horizontal distance to the goal position
        """
        soft_area = self.distance < 1.25
        tight_area = self.distance < 0.4
        dy_error = torch.abs(self.P_base[:, 1])
        return torch.exp(-dy_error / 0.005) * tight_area + 0.1 * torch.exp(-dy_error/0.1) * soft_area
    
    def _reward_forward(self):
        # penalize stationary in grasp area
        vel_sigma = 0.2
        soft_area = torch.logical_and(self.distance < 2.0, self.distance > 0.5)
        forward_soft = self.base_lin_vel[:, 0] > 0.3
        tight_area = self.distance < 0.4
        slow_approach = torch.logical_and(self.base_lin_vel[:, 0] < 0.5, self.base_lin_vel[:, 0] > vel_sigma)  # robot should slow down when approaching the target
        slow_approach = torch.logical_and(slow_approach, self.base_ang_vel[:, 2].abs() < 0.3)  # also reduce angular velocity
        tracking_velocity = torch.exp(-(self.base_lin_vel[:, 0] - vel_sigma) / 0.1) * (self.base_lin_vel[:, 0] > vel_sigma)
        pitch_restricted = torch.logical_and(self.euler_rpy[:, 1] > 0.25, self.object_pos[:, 2] < 0.2)
        return 1.0 * soft_area * forward_soft.float() * (~self.physically_out_of_view).float() + 10 * tight_area * slow_approach * tracking_velocity * pitch_restricted.float()

    def _reward_reach_grasp_area(self):
        target_grasp_width = 0.04
        grasp_area = (torch.abs(self.P_base[:, 1]) < (target_grasp_width/2.0))
        reach_grasp_distance = (self.distance < 0.6)
        slow_approach = torch.logical_and(self.base_lin_vel[:, 0] < 0.35, self.base_lin_vel[:, 0] > 0.1)  # robot should slow down when approaching the target
        return reach_grasp_distance * grasp_area.float() * slow_approach.float()
    
    # PCA-based Rewards
    
    def _reward_approach_tip(self):
        """ Reward for approaching the closest tip (Head or Tail) instead of center. """
        # self.sigma_points_base: [N, 5, 3]
        # 0: Center, 1: Head, 2: Tail
            
        p_head_diff = self.p_head_world[:, :2] - self.gripper_world[:, :2]
        p_tail_diff = self.p_tail_world[:, :2] - self.gripper_world[:, :2]
        
        # Distance to head and tail (in base frame, robot is at 0,0,0)
        dist_head = torch.norm(p_head_diff, dim=-1)
        dist_tail = torch.norm(p_tail_diff, dim=-1)
        
        # Min distance
        min_dist = torch.min(dist_head, dist_tail)

        near_object = (min_dist < 1.0).float()
        
        return near_object * (1.0 / (1.0 + 10 * torch.square(min_dist)))

    def _reward_conditional_alignment(self):
        """
        Conditional Alignment Reward:
        If object is long (max_dim > gripper_width), reward aligning with the long axis.
        """
        sigma = 0.1
        
        # Use pre-computed long axis and condition
        # self.axis_long_world: [N, 3]
        # self.obj_is_too_long: [N]
        
        dot_prod = torch.abs(torch.sum(self.forward_world * self.axis_long_world, dim=-1))
        
        # Reward: 1.0 when aligned (dot=1), 0.0 when perpendicular (dot=0)
        reward = self.obj_is_too_long.float() * torch.exp(-(1.0 - dot_prod) / sigma)
        
        return reward

    def _reward_conditional_perpendicular_penalty(self):
        """
        Conditional Perpendicular Penalty (Anti-proposition):
        If object is long, penalize being perpendicular to the long axis.
        """
        sigma = 0.1
        
        dot_prod = torch.abs(torch.sum(self.forward_world * self.axis_long_world, dim=-1))
        
        # Penalty: High when perpendicular (dot=0), Low when aligned (dot=1)
        penalty = self.obj_is_too_long.float() * torch.exp(-dot_prod / sigma)
        
        return penalty

    def _reward_target_directed_velocity(self):
        """
        Reward robot for aligning its velocity vector towards the nearest graspable endpoint (Short Edge Vertex).
        Target-Directed Velocity Reward.
        """
        # 1. Prepare Data
        robot_pos = self.root_states[:, :3]
        robot_vel = self.root_states[:, 7:10] # World frame linear velocity
        object_pos = self.object_pos
                
        # 3. Dynamic Target Selection (Nearest Endpoint)
        dist_to_head = torch.norm(self.p_head_world - robot_pos, dim=-1)
        dist_to_tail = torch.norm(self.p_tail_world - robot_pos, dim=-1)
        
        # Choose mask (1 if head is closer)
        choose_head = (dist_to_head < dist_to_tail).float().unsqueeze(-1)
        
        # Endpoint target
        p_endpoint = choose_head * self.p_head_world + (1.0 - choose_head) * self.p_tail_world
        
        # 4. Final Target
        # If too long -> p_endpoint, else -> object_pos
        p_target = torch.where(self.obj_is_too_long.unsqueeze(-1), p_endpoint, object_pos)
        
        # 5. Expected Direction Vector
        vec_to_target = p_target - robot_pos
        dir_to_target = vec_to_target / (torch.norm(vec_to_target, dim=-1, keepdim=True) + 1e-6)
        
        # 6. Velocity Projection
        vel_projection = torch.sum(robot_vel * dir_to_target, dim=-1)
        
        # 7. Reward Shaping
        target_speed = 0.5
        r_vel = torch.exp(-torch.square(vel_projection - target_speed))
        
        return r_vel

    def _reward_missing_sigma_points(self):
        """ Reward for missing sigma points in the view
        """
        return (~self.is_valid).float()
    
    def _reward_visual_foreshortening(self):
        """
        [Visual Foreshortening Reward]
        Encourage the robot to adjust its pose so that the "lateral span" of the object in the robot's field of view is minimized.
        When the robot is facing the endpoint of a long object, the projected lateral span (Y-axis variance) is minimized.
        """
        # 1. Get the 5 Sigma Points in Robot Body Frame
        # self.sigma_points_base: [N, 5, 3]
        points_body = self.sigma_points_base
        
        # 2. Extract lateral coordinates (Y axis)
        # We want these 5 points to be clustered on the Y axis, meaning the robot is facing the long axis
        ys = points_body[..., 1] # (N, 5)
        
        # 3. Calculate lateral span (Spread)
        # Option A: Standard Deviation - Recommended, smoother
        y_std = torch.std(ys, dim=-1) # (N,)
        
        # We want y_std to be as small as possible.
        # Set an "ideal convergence threshold" determined by object width
        target_std = 0.01 
        
        # Use Gaussian kernel: higher reward as it approaches 0
        r_foreshortening = torch.exp(-torch.square(y_std - target_std) / 0.005)
        
        # Apply mask
        return r_foreshortening * self.obj_is_too_long.float()

    def _reward_visual_foreshortening_2d(self):
        """
        Reward for minimizing the horizontal spread of sigma points relative to vertical spread in 2D image.
        Encourages facing the short side of the object (foreshortening).
        """
        # self.sigma_points_2d: [N, 5, 2] (Normalized u, v)
        
        # Scale by image dimensions to get pixel-like spread
        # self.cam_img_w, self.cam_img_h are tensors [N, 1]
        u = self.sigma_points_2d[..., 0] * self.cam_img_w
        v = self.sigma_points_2d[..., 1] * self.cam_img_h
        
        # Calculate spread (Standard Deviation)
        std_u = torch.std(u, dim=-1) # [N]
        std_v = torch.std(v, dim=-1) # [N]
        
        # Metric: Ratio of horizontal spread to total spread
        # If std_u is large (horizontal line), metric -> 1.0
        # If std_u is small (vertical line or point), metric -> 0.0
        metric = std_u / (std_u + std_v + 1e-5)
        
        # We want to minimize metric.
        # Reward = 1.0 - metric
        reward = (1.0 - metric)
        
        # Only apply if object is "long"
        reward = reward * self.obj_is_too_long.float()
        
        return reward

    def _reward_optimal_pose_tracking(self):
        """
        Reward for tracking the optimal grasp pose (Position & Orientation) AND approaching velocity.
        """
        # 1. Position Error (Decomposed)
        target_pos = self.optimal_grasp_pos
        # current_pos = self.root_states[:, :3]
        # [Correction] Use Gripper Position
        current_pos = self.gripper_world
        target_dir = self.optimal_approach_dir
        
        pos_diff = current_pos - target_pos
        pos_diff[:, 2] = 0.0 # Explicitly ignore Z error as requested
        
        # Longitudinal Error (Along the approach line)
        # Positive means closer to object (past the target point), Negative means behind
        long_dist = torch.sum(pos_diff * target_dir, dim=-1)
        
        # Lateral Error (Perpendicular to approach line)
        # diff_perp = diff - long * dir
        lat_diff = pos_diff - long_dist.unsqueeze(-1) * target_dir
        lat_error_sq = torch.sum(torch.square(lat_diff), dim=-1)
        
        # [CRITICAL] One-sided Longitudinal Penalty
        # We only penalize if we are BEHIND the target point (long_dist < 0).
        # If we are past it (long_dist > 0), we treat longitudinal error as 0.
        # This allows the robot to move through the target point towards the object without penalty.
        long_error_sq = torch.square(torch.clamp(long_dist, max=0.0))
        
        # Combine errors (Lateral is critical, Longitudinal is for catching up)
        pos_error_sq = lat_error_sq + long_error_sq
        r_pos = torch.exp(-pos_error_sq / 0.1) # sigma ~ 0.3m
        
        # 2. Orientation Error (Weighted Euler)
        # User wants Yaw important, Pitch easy, Roll not important.
        q_robot = self.root_states[:, 3:7]
        q_target = self.optimal_grasp_quat
        
        roll_r, pitch_r, yaw_r = get_euler_xyz(q_robot)
        roll_t, pitch_t, yaw_t = get_euler_xyz(q_target)
        
        roll_err = torch.abs(wrap_to_pi(roll_r - roll_t))
        pitch_err = torch.abs(wrap_to_pi(pitch_r - pitch_t))
        yaw_err = torch.abs(wrap_to_pi(yaw_r - yaw_t))
        
        # Weights: Yaw=1.0, Pitch=0.1, Roll=0.0 (Ignore roll)
        w_yaw = 1.0
        w_pitch = 0.1
        w_roll = 0.0
        rot_error_sq = w_yaw * torch.square(yaw_err) + w_pitch * torch.square(pitch_err) + w_roll * torch.square(roll_err)
        
        r_rot = torch.exp(-rot_error_sq / 0.1) # sigma ~ 0.3 rad
        
        # 3. Velocity Tracking (Forward along approach axis)
        # Target speed: 0.2 m/s
        # v_robot = self.root_states[:, 7:10]
        # v_proj = torch.sum(v_robot * target_dir, dim=-1)
        # v_error_sq = torch.square(v_proj - 0.15)
        
        # r_vel = torch.exp(-v_error_sq / 0.04) # sigma = 0.2 m/s

        # go_forward = (v_proj > 0.1).float()

        
        return r_pos * r_rot

    def _reward_successful_grasp(self):
        """
        Reward for being in the success state (reaching object center with correct alignment).
        Given continuously while the robot maintains the state.
        """
        r_pos = torch.exp(-(self.dist_to_opt) / 0.1) # sigma ~ 0.14 m
        r_rot = torch.exp(-(self.yaw_err) / 0.1) # sigma ~ 0.2 rad
        return self.is_success_state.float() * (1.0 + 10 * r_pos +  1 * r_rot)
    
    def _reward_backup(self):
        """ Reward for backing up when too close to the object
        """
        too_close = self.distance < 0.5
        backing_up = self.base_lin_vel[:, 0] < -0.1
        # return too_close.float() * backing_up.float()
        return backing_up.float()
    
    def _reward_sequential_reaching(self):
        """ Reward for reaching Hint Pose first, then Optimal Pose.
            Uses a path-following approach with strict gating to enforce sequence:
            1. Align & Reach Hint (Path Reward + Align Reward)
            2. Move to Optimal (Goal Reward, gated by Path & Align)
        """
        # Important: We only consider 2D positions (X, Y), ignore Z.
        hint_pos_2d = self.hint_pos[:, :2]
        optimal_pos_2d = self.optimal_grasp_pos[:, :2]
        gripper_pos_2d = self.gripper_world[:, :2]

        # Vector from Hint to Optimal
        v_path_2d = optimal_pos_2d - hint_pos_2d # [N, 2]
        len_sq = torch.sum(v_path_2d**2, dim=1, keepdim=True) # [N, 1]
        
        # Vector from Hint to Gripper
        v_gripper_2d = gripper_pos_2d - hint_pos_2d # [N, 2]
        
        # Project Robot onto Line (t * v_path)
        # t = (v_robot . v_path) / len_sq
        t = torch.sum(v_gripper_2d * v_path_2d, dim=1, keepdim=True) / (len_sq + 1e-6)
        t_clamped = torch.clamp(t, 0.0, 1.0)
        
        # Closest point on segment
        p_closest_2d = hint_pos_2d + t_clamped * v_path_2d
        
        # Distance to Path (Cross Track Error)
        d_path = torch.norm(gripper_pos_2d - p_closest_2d, dim=1)
        
        # Distance to Optimal (Goal Error)
        d_goal = torch.norm(gripper_pos_2d - optimal_pos_2d, dim=1)
        
        # Orientation Error
        q_robot = self.root_states[:, 3:7]
        q_target = self.optimal_grasp_quat
        _, _, yaw_r = get_euler_xyz(q_robot)
        _, _, yaw_t = get_euler_xyz(q_target)
        yaw_err = torch.abs(wrap_to_pi(yaw_r - yaw_t))
        
        # Rewards
        # 1. Path Following Reward (Pulls to Hint if t<0, then keeps on line)
        # Strict corridor (5cm)
        r_path = torch.exp(-d_path / 0.02)
        
        # 2. Orientation Reward
        # Strict alignment (0.1 rad ~ 5.7 deg)
        r_align = torch.exp(-yaw_err / 0.1)
        
        # 3. Goal Reaching Reward (Pulls along line to Optimal)
        r_goal = torch.exp(-d_goal / 0.05)
        
        # Combined Reward:
        # r_base = r_path * r_align (Must be on path AND aligned)
        # Total = r_base * (1 + r_goal)
        # If misaligned or off-path, r_base -> 0, so Total -> 0.
        # If aligned and on path, Total -> 1 + r_goal (Max 2).
        return r_path * r_align * (1.0 + 1 * r_goal)