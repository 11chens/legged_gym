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
        self.sigma = 0.33
        self.camera_sensor = CameraSensor(
            batch_size=self.num_envs, 
            cfg=self.cfg.camera_sensor,
            device=self.device,
            )
        
        self.horizontal_fov_rad = self.camera_sensor.horizontal_fov / 180.0 * np.pi
        self.horizontal_fov_half_rad = self.horizontal_fov_rad / 2.0
        self.vertical_fov_rad = self.camera_sensor.vertical_fov / 180.0 * np.pi
        self.vertical_fov_half_rad = self.vertical_fov_rad / 2.0

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
        
        if target_cfg.shape.type == "mixed":
            self.shape_types_list = target_cfg.shape.types
            self.env_shape_type_indices = torch.randint(0, len(self.shape_types_list), (self.num_envs,), device=self.device)
        else:
            self.shape_types_list = [target_cfg.shape.type]
            self.env_shape_type_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Randomize props for all envs
        self._randomize_object_props(torch.arange(self.num_envs, device=self.device))
        
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
        
        # Object State
        self.object_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.object_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.object_quat[:, 3] = 1.0 # Identity
        
        # Visualization Utils
        self.vis_utils = VisualizationUtils(self)
        
        # Sigma Points in Base Frame (for Obs)
        self.sigma_points_base = torch.zeros(self.num_envs, 5, 3, device=self.device)

        # Cache config parameters
        self.target_cfg = self.cfg.target
        self.perception_frame = self.target_cfg.perception.frame
        self.use_geometric_weight = self.target_cfg.perception.use_geometric_weight
        self.randomize_orientation = self.target_cfg.init.randomize_orientation
        self.debug_timer = self.target_cfg.perception.debug_timer
        self.debug_info = self.target_cfg.perception.debug_info

        # Cache shape params for resampling
        self.shape_type = self.target_cfg.shape.type
        if self.shape_type == "cylinder" or self.shape_type == "sphere":
             self.z_offset = self.target_cfg.shape.radius
        else:
             self.z_offset = self.target_cfg.shape.dims[2] / 2.0
             
        # Cache camera drift params
        self.enable_out_of_view_drift = self.cfg.camera_sensor.enable_out_of_view_drift
        self.drift_scale = self.cfg.camera_sensor.drift_scale
        self.max_out_of_view_duration = self.cfg.camera_sensor.max_out_of_view_duration
        self.save_debug_images = self.cfg.camera_sensor.save_debug_images
        self.vis_target_points = self.cfg.camera_sensor.vis_target_points
        self.vis_sigma_3d = self.cfg.camera_sensor.vis_sigma_3d
        self.vis_sigma_2d = self.cfg.camera_sensor.vis_sigma_2d

        # Pre-allocate tensors for resampling
        self.t_min_default = torch.ones((self.num_envs, 1), device=self.device) * 1.0
        self.t_max_default = torch.ones((self.num_envs, 1), device=self.device) * 3.5
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

        self.x_axis_local = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self.y_axis_local = torch.tensor([0.0, 1.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self.z_axis_local = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)

        self.gripper_width = 0.08

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
        self.out_of_view_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.out_of_view_drift = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        
        # [NEW] 3D Drift for Sigma Points
        self.sigma_drift = torch.zeros(self.num_envs, 5, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_valid_sigma_points = torch.zeros(self.num_envs, 5, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.physically_out_of_view = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)

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

        self.camera_noise_vec = self._get_camera_noise_vec()
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
        self._resample_commands(env_ids)

        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_root_vel[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.object_moved[env_ids] = False
        self.out_of_view_timer[env_ids] = 0.
        self.out_of_view_drift[env_ids] = 0.
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

        self._post_physics_step_callback() # [new] update nav_commands

        # compute observations, rewards, resets, ...
        self.check_termination() # time out or collision
        self.compute_reward() # [new] add goal-reaching reward
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids) # [new] terrain_cur(env_origin), reset robot, resample_cmds
        self.update_image(env_ids=0) # update env0 image for debug viz, instead of all envs
        self.compute_observations() # [new] update nav_commands for nav policy

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
        """ Resample object pose once per episode when in specified range
        """
        in_range = (self.distance > 0.5) & (self.distance < 1.0)
        to_move = in_range & (~self.object_moved)
        env_ids_resample = to_move.nonzero(as_tuple=False).flatten()
        if len(env_ids_resample) > 0:
            self._resample_commands(env_ids_resample, on_the_way=True)
            self.object_moved[env_ids_resample] = True
    
    def _compute_long_end_points_in_world(self):
        """ Compute the long axis endpoints (head & tail) in world frame
        """
        # Find the index of the longest dimension
        # dims: [N, 3]
        dims = self.tracker.object_dims
        max_dim_vals, max_dim_inds = torch.max(dims, dim=1) # [N], [N]
        
        # Select the corresponding axis vector
        # We have x_axis_world, y_axis_world, z_axis_world: [N, 3]
        # Stack them: [N, 3, 3]
        all_axes = torch.stack([self.x_axis_world, self.y_axis_world, self.z_axis_world], dim=1)
        
        # Gather the axis corresponding to max_dim_inds
        # Create batch indices
        batch_indices = torch.arange(self.num_envs, device=self.device)
        self.axis_long_world = all_axes[batch_indices, max_dim_inds] # [N, 3]
        
        half_length = (max_dim_vals / 2.0).unsqueeze(-1)
        self.p_head_world = self.object_pos + self.axis_long_world * half_length
        self.p_tail_world = self.object_pos - self.axis_long_world * half_length
                    
        # Determine if object is too long (Check Max Dimension)
        self.obj_is_too_long = (max_dim_vals > self.gripper_width)

    def _post_physics_step_callback(self):
        """ origin: resample_cmds[env_ids] at resampling_time
            new: update nav_commands every step
        """
        if self.cfg.commands.resample_on_the_way:
            self.resample_object_pose_on_the_way()

        # update distance and reach_goal
        self.distance = torch.norm(self.root_states[:, :2] - self.object_pos[:, :2], dim=1)
        self.reach_goal = self.distance < (self.sigma)

        # get self.forward_world, self.x_axis_world, self.y_axis_world, self.z_axis_world
        self._compute_robot_object_directions()

        # get self.camera_transform_world('R': self.R_world_to_cam, 'T': self.camera_world)
        self._compute_camera_pose_in_world()

        # map object center to image plane and get depth_in_view
        self._compute_object_center_map()

        # get long end points of object in world frame (self.p_head_world, self.p_tail_world)
        self._compute_long_end_points_in_world() # nontrivial

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
    
    def _randomize_object_props(self, env_ids):
        if len(env_ids) == 0:
            return

        target_cfg = self.cfg.target
        
        # 1. Determine Shape Types
        if target_cfg.shape.type == "mixed":
            type_indices = torch.randint(0, len(self.shape_types_list), (len(env_ids),), device=self.device)
            self.env_shape_type_indices[env_ids] = type_indices
        
        # 2. Iterate over types
        for i, type_name in enumerate(self.shape_types_list):
            subset_indices = self.env_shape_type_indices[env_ids]
            mask = (subset_indices == i)
            
            if not mask.any():
                continue
            
            count = mask.sum().item()
            
            # Generate random params
            if type_name == "cylinder":
                r = torch_rand_float(target_cfg.shape.radius_range[0], target_cfg.shape.radius_range[1], (count, 1), device=self.device)
                h = torch_rand_float(target_cfg.shape.height_range[0], target_cfg.shape.height_range[1], (count, 1), device=self.device)
                params = torch.cat([r, h], dim=1)
            elif type_name == "cuboid":
                x = torch_rand_float(target_cfg.shape.dims_range[0][0], target_cfg.shape.dims_range[0][1], (count, 1), device=self.device)
                y = torch_rand_float(target_cfg.shape.dims_range[1][0], target_cfg.shape.dims_range[1][1], (count, 1), device=self.device)
                z = torch_rand_float(target_cfg.shape.dims_range[2][0], target_cfg.shape.dims_range[2][1], (count, 1), device=self.device)
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
            global_indices = env_ids[mask]
            self.object_dims[global_indices] = dims
            self.local_points[global_indices] = pts
            self.local_normals[global_indices] = nrms
            
        # Update tracker if it exists
        if hasattr(self, 'tracker'):
            self.tracker.object_dims[env_ids] = self.object_dims[env_ids]
            self.tracker.local_points[env_ids] = self.local_points[env_ids]
            self.tracker.local_normals[env_ids] = self.local_normals[env_ids]

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
        P_world[:, 2] = self.z_offset

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
        
        # 3. Apply shape-specific base rotation (e.g. Cylinder needs to lie down)
        q_final = q_yaw.clone()
        
        # Check for cylinder
        is_cylinder = False
        cylinder_mask = None
        
        if self.shape_type == "cylinder":
            is_cylinder = True
            cylinder_mask = slice(None)
        elif self.shape_type == "mixed":
            if "cylinder" in self.shape_types_list:
                cyl_idx = self.shape_types_list.index("cylinder")
                current_types = self.env_shape_type_indices[env_ids]
                cylinder_mask = (current_types == cyl_idx)
                if cylinder_mask.any():
                    is_cylinder = True
        
        if is_cylinder:
            # Rotate 90 deg around Y axis to make Z-axis cylinder lie along X-axis
            val = np.sin(np.pi / 4)
            q_lay = torch.tensor([0.0, val, 0.0, val], device=self.device, dtype=torch.float).view(1, 4)
            
            if self.shape_type == "cylinder":
                q_final = quat_mul(q_yaw, q_lay.repeat(len(env_ids), 1))
            elif self.shape_type == "mixed":
                q_yaw_masked = q_yaw[cylinder_mask]
                q_lay_masked = q_lay.repeat(q_yaw_masked.shape[0], 1)
                q_final[cylinder_mask] = quat_mul(q_yaw_masked, q_lay_masked)

        return q_final
    
    def _resample_commands(self, env_ids, on_the_way=False):
        """ set the goal position in the visualable zone of the camera
        """
        if len(env_ids) == 0:
            return
        
        if not on_the_way:
            # Randomize Object Properties (Dims, Points)
            self._randomize_object_props(env_ids)
        
        # 1. Resample Object Position
        P_world = self._resample_object_positions(env_ids)
        # 2. Resample Object Orientation
        q_final = self._resample_object_orientations(env_ids)

        self.object_pos[env_ids] = P_world
        self.object_quat[env_ids] = q_final
    
    def _compute_robot_object_directions(self):
        """ Compute robot forward vector and object axes in world frame.
        """
        # Robot forward vector in world frame
        self.forward_world = quat_apply(self.base_quat, self.forward_vec)
        # Object axes in world frame
        self.x_axis_world = quat_apply(self.object_quat, self.x_axis_local)
        self.y_axis_world = quat_apply(self.object_quat, self.y_axis_local)
        self.z_axis_world = quat_apply(self.object_quat, self.z_axis_local)
        
        # Compute Head and Tail Points (Endpoints along X-axis)
        half_length = (self.tracker.object_dims[:, 0] / 2.0).unsqueeze(-1)
        self.p_head_world = self.object_pos + self.x_axis_world * half_length
        self.p_tail_world = self.object_pos - self.x_axis_world * half_length
        
    def _compute_sigma_points_base(self, sigma_points_3d, base_pos, base_quat, is_valid):
        """ Compute perception features (Sigma Points) in Robot Base Frame.
        """
        delta = sigma_points_3d - base_pos # [N, 5, 3]
        sigma_points_base_flat = quat_rotate_inverse(base_quat.expand(-1, 5, -1).reshape(-1, 4), delta.reshape(-1, 3))
        current_sigma_base = sigma_points_base_flat.view(self.num_envs, 5, 3)
        
        # If valid, update buffer. If invalid, set invalid data
        valid_mask = is_valid.view(self.num_envs, 1, 1)
        invalid_sigma = torch.ones_like(current_sigma_base) * -1.0
        self.sigma_points_base = torch.where(valid_mask, current_sigma_base, invalid_sigma)

    def _compute_sigma_points_camera(self):
        """ Compute perception features (Sigma Points) in Camera Frame.
        """
        delta_cam = self.sigma_points_base - self.camera_sensor.T.unsqueeze(1)  # [N, 5, 3]
        res = torch.bmm(self.camera_sensor.R, delta_cam.transpose(1, 2)) # [N, 3, 5]
        self.sigma_points_camera = res.transpose(1, 2) # [N, 5, 3] (x, y, z)
    
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
        
        self.sigma_points_image = torch.stack([u_norm, v_norm, z], dim=-1) # [N, 5, 3]

    def _get_nav_commands(self):
        """ Compute perception features (Sigma Points) and update nav_commands.
        """
        # Robot State: self.root_states [N, 13] (pos: 0-3, quat: 3-7)
        base_pos = self.root_states[:, :3].unsqueeze(1) # [N, 1, 3]
        base_quat = self.root_states[:, 3:7].unsqueeze(1) # [N, 1, 4]

        # 1. Run Perception Tracker to get 3D Sigma Points in World Frame
        # sigma_points_2d: [N, 5, 2] in Image Plane
        # sigma_points_3d: [N, 5, 3] in World Frame
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

        self._compute_sigma_points_base(sigma_points_3d, base_pos, base_quat, is_valid)
 
        # 2. Transform to Configured Frame
        if self.perception_frame == "base":
            self._compute_sigma_points_base(sigma_points_3d, base_pos, base_quat, is_valid)
            self.sigma_points_obs = self.sigma_points_base
        elif self.perception_frame == "camera":
            self._compute_sigma_points_camera()
            self.sigma_points_obs = self.sigma_points_camera
        elif self.perception_frame == "image":
            if sigma_points_2d is not None:
                self.sigma_points_image = sigma_points_2d
            else:
                self._compute_sigma_points_camera()
                self._compute_sigma_points_image()
            self.sigma_points_obs = self.sigma_points_image
        else:
            raise ValueError(f"Unknown perception frame: {self.perception_frame}")

        # 3. Update nav_commands (15 dims)
        # [p_center, p_head, p_tail, p_side1, p_side2]
        self.nav_commands = self.sigma_points_obs.reshape(self.num_envs, 15)
        
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

            # Downsample target points for visualization
            stride_size = int(self.tracker.num_points // self.cfg.camera_sensor.num_vis_points)
            points_local_sample = self.tracker.local_points[0][::stride_size]  # [num_vis_points, 3]
            # Transform to world frame
            points_world_sample = quat_apply(self.object_quat[0].unsqueeze(0).expand(points_local_sample.shape[0], -1), points_local_sample) + self.object_pos[0]
            # Draw target points using cross markers
            self.vis_utils.draw_3d_lines(points_world_sample, color=[0, 0, 1], env_idx=0)
            
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
            
            # 3. Project to camera image plane and Draw
            if self.common_step_counter % 10 == 0:
                points_dict = {}
                if self.vis_target_points:
                    points_dict["target"] = points_world_sample
                if self.vis_sigma_3d:
                    points_dict["sigma_3d"] = sigma_points_3d[0]
                
                points_2d_dict = {}
                if self.vis_sigma_2d and sigma_points_2d is not None:
                    points_2d_dict = {
                        "sigma_2d": sigma_points_2d[0]
                    }
                
                self.vis_utils.draw_2d_image(
                    points_3d_dict=points_dict, 
                    camera_sensor=self.camera_sensor, 
                    env_idx=0, 
                    filename=f"debug_cam_{self.common_step_counter}.png",
                    save_images=self.save_debug_images,
                    points_2d_dict=points_2d_dict
                )

    
    def _get_camera_noise_vec(self):
        noise_vec = torch.zeros_like(self.nav_commands[0])
        noise_scales = self.cfg.noise.noise_scales
        self.add_camera_noise = self.cfg.noise.add_camera_noise
        noise_vec[0] = noise_scales.P_img_u
        noise_vec[1] = noise_scales.P_img_v
        noise_vec[2] = noise_scales.P_img_depth
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
        noise_vec[start:end] = 0. # self.nav_commands (get from camera, noise already added)

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
        # total obs: 3 + 3 + 3 + 1 + 1 + 4 = 15
        obs_buf = torch.cat([
            self.base_lin_vel_pred * self.obs_scales.lin_vel, # 3
            self.base_ang_vel * self.obs_scales.ang_vel, # 3
            self.projected_gravity, # 3
            self.euler_rpy[:, :3] * self.obs_scales.euler_rpy,  # rpy 3
            self.delay_nav_commands, # 3
            self.nav_actions # 4
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
            # DEBUG SHAPES
            s_sigma = self.sigma_points_obs.reshape(self.num_envs, -1).shape[1]
            s_hist = self.obs_hist_buffer.view(self.num_envs, -1).shape[1]
            s_delay = self.delay_nav_commands_hist_buffer.view(self.num_envs, -1).shape[1]
            total = s_sigma + s_hist + s_delay
            if total != 320:
                 print(f"SHAPE MISMATCH: sigma={s_sigma}, hist={s_hist}, delay={s_delay}, total={total}")
                 raise RuntimeError(f"Observation shape mismatch: {total} vs 320")

            self.obs_buf = torch.cat([
                self.sigma_points_obs.reshape(self.num_envs, -1), self.obs_hist_buffer.view(self.num_envs, -1), self.delay_nav_commands_hist_buffer.view(self.num_envs, -1)
                ], dim=-1)


        # if hasattr(self, 'image'):
        #     # image = self.image.detach().cpu().numpy()
        #     # self.camera_sensor.visualize_img(image)
        #     self._draw_camera_position()

    def _draw_camera_position(self):
        sphere_red = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(1, 0, 1))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3])
            camera_points = quat_apply(self.base_quat[i], self.camera_position_scalar)  # camera position in base frame
            x = camera_points[0] + base_pos[0]
            y = camera_points[1] + base_pos[1]
            z = camera_points[2] + base_pos[2]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_red, self.gym, self.viewer, self.envs[i], sphere_pose) 
    
    def _draw_fov(self, env_idx=0):
        """ Draw FOV lines for debugging
        """
        # Only draw for the first environment
        cam = self.camera_sensor
        
        # Get camera parameters for env 0
        def get_param(param, idx):
            if isinstance(param, torch.Tensor):
                return param[idx].item()
            return param
            
        fx = get_param(cam.fx, env_idx)
        fy = get_param(cam.fy, env_idx)
        cx = get_param(cam.cx, env_idx)
        cy = get_param(cam.cy, env_idx)
        
        img_w = get_param(cam.img_width, env_idx)
        img_h = get_param(cam.img_height, env_idx)
        
        # Define corners in image pixel coordinates: Top-Left, Top-Right, Bottom-Right, Bottom-Left
        corners_pix = torch.tensor([
            [0, 0],
            [img_w, 0],
            [img_w, img_h],
            [0, img_h]
        ], device=self.device, dtype=torch.float)
        
        # Transform to Camera Frame
        depth = 3.5 # Max visual distance
        
        # x = (u - cx) * Z / fx, y = (v - cy) * Z / fy, z = Z
        corners_cam_x = (corners_pix[:, 0] - cx) * depth / fx
        corners_cam_y = (corners_pix[:, 1] - cy) * depth / fy
        corners_cam_z = torch.full((4,), depth, device=self.device)
        
        corners_cam = torch.stack([corners_cam_x, corners_cam_y, corners_cam_z], dim=-1) # (4, 3)
        
        # Transform to Base Frame: P_base = R^T * P_cam + T
        R = cam.R[env_idx] # (3, 3)
        T = cam.T[env_idx] # (3)
        corners_base = torch.matmul(corners_cam, R) + T
        center_base = T
        
        # Transform to World Frame
        base_pos = self.root_states[env_idx, :3]
        base_quat = self.base_quat[env_idx]
        
        corners_world = quat_apply(base_quat.repeat(4, 1), corners_base) + base_pos
        center_world = quat_apply(base_quat, center_base) + base_pos
        
        # Draw lines using spheres for thickness
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 0, 1))
        
        corners = corners_world.cpu().numpy()
        center = center_world.cpu().numpy()
        
        def draw_thick_line(start, end, num_spheres=50):
            for k in range(num_spheres + 1):
                t = k / num_spheres
                pos = start + (end - start) * t
                pose = gymapi.Transform(gymapi.Vec3(pos[0], pos[1], pos[2]), r=None)
                # Use None for env to draw in world coordinates
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, None, pose)

        # 4 lines from center to corners
        for k in range(4):
            draw_thick_line(center, corners[k], num_spheres=50)
            
        # 4 lines connecting corners
        for k in range(4):
            draw_thick_line(corners[k], corners[(k+1)%4], num_spheres=50)

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

    def _reward_reach_target(self):
        # return (1. /(1. + 100*torch.square(self.distance)))
        return torch.exp(-self.distance/self.cfg.rewards.tracking_sigma)
    
    def _reward_stand_still(self):
        actions_norm = torch.norm(torch.abs(self.nav_actions[:, :3]), dim=-1) 
        return (1. /(1. + 10*torch.square(actions_norm))) * (self.reach_goal)
    
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

    def _reward_horizontal_distance_error(self):
        # Penalize large horizontal distance error
        target_grasp_width = 0.1
        dy_error = torch.abs(self.P_base[:, 1]) - target_grasp_width
        dy_error = torch.clip(dy_error, min=0.0)
        return torch.square(dy_error)
    
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
    
    # [NEW] PCA-based Rewards
    
    def _reward_approach_tip(self):
        """ Reward for approaching the closest tip (Head or Tail) instead of center. """
        # self.sigma_points_base: [N, 5, 3]
        # 0: Center, 1: Head, 2: Tail
            
        p_head_diff = self.p_head_world[:, :2] - self.root_states[:, :2]
        p_tail_diff = self.p_tail_world[:, :2] - self.root_states[:, :2]
        
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
    
