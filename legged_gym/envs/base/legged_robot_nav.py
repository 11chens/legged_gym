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
from legged_gym.utils.torch_math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float, yaw_quat, cart2polar, quat_to_rot_matrix
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.go2.go2_nav_config import Go2NavFlatCfg
from .legged_robot import LeggedRobot
from legged_gym.utils.camera_sensor import CameraSensor
from legged_gym.perception.tracker import PCATargetTracker
from legged_gym.utils.visualization import VisualizationUtils
from legged_gym.envs.base.object_manager import ObjectManager
from legged_gym.envs.base.grasp_pose_optimizer import GraspPoseOptimizer

class LeggedRobotNav(LeggedRobot):
    cfg : Go2NavFlatCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        self.debug_viz = self.cfg.debug_viz
    
        # Gripper Config
        self.gripper_width = self.cfg.gripper.gripper_width
        self.gripper_offset = torch.tensor(self.cfg.gripper.gripper_offset, device=self.device).repeat(self.num_envs, 1)

        # Initialize Perception Tracker
        target_cfg = self.cfg.target
        
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
        self.use_geometric_weight = self.target_cfg.perception.use_geometric_weight
        self.randomize_orientation = self.target_cfg.init.randomize_orientation
        self.debug_timer = self.target_cfg.perception.debug_timer
        self.debug_info = self.target_cfg.perception.debug_info
        self.save_debug_images = self.cfg.camera_sensor.save_debug_images
        
        # Ranges
        self.dist_fwd_range = self.cfg.commands.resample.ranges.dist_fwd
        self.dist_lat_range = self.cfg.commands.resample.ranges.dist_lat
        
    def _init_buffers(self):
        """ inherit loco vars: self.commands[vx, vy, vyaw, pitch], self.actons[joint_pos]
            add nav vars: self.nav_commands[theta, rho], self.nav_actions[vx, vy, vyaw] = self.commands[:, :3]
            update vars: self.obs_buf -> nav_polciy
        """
        super()._init_buffers()

        # Initialize Camera Sensor & Params
        self.camera_sensor = CameraSensor(
            batch_size=self.num_envs, 
            cfg=self.cfg.camera_sensor,
            device=self.device,
        )

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

        self.camera_params_dict = {
            'fx': self.cam_fx,
            'fy': self.cam_fy,
            'cx': self.cam_cx,
            'cy': self.cam_cy,
            'img_width': self.cam_img_w,
            'img_height': self.cam_img_h,
            'img_w': self.cam_img_w,
            'img_h': self.cam_img_h
        }
        
        self.camera_transform_dict = {
            'R': self.camera_sensor.R,
            'T': self.camera_sensor.T
        }

        # Initialize Object Manager
        self.gripper_width = self.cfg.gripper.gripper_width
        self.object_manager = ObjectManager(
            cfg=self.cfg,
            device=self.device,
            num_envs=self.num_envs,
            gripper_width=self.gripper_width,
            camera_params_dict=self.camera_params_dict
        )
        self.shapes = self.object_manager.shapes
        self.ycb_manager = self.object_manager.ycb_manager

        # Link attributes from ObjectManager (In-place buffers)
        self.object_dims = self.object_manager.object_dims
        self.ycb_indices = self.object_manager.ycb_indices
        self.local_points = self.object_manager.local_points
        self.local_normals = self.object_manager.local_normals
        self.object_pos = self.object_manager.object_pos
        self.object_quat = self.object_manager.object_quat
        self.x_axis_local = self.object_manager.x_axis_local
        self.y_axis_local = self.object_manager.y_axis_local
        self.z_axis_local = self.object_manager.z_axis_local
        self.shape_types_list = self.object_manager.shape_types_list
        self.env_shape_type_indices = self.object_manager.env_shape_type_indices
        
        # Link attributes that were in _init_buffers
        self.obj_is_vertical = self.object_manager.obj_is_vertical
        self.obj_is_too_long = self.object_manager.obj_is_too_long
        self.object_z_offset = self.object_manager.object_z_offset
        self.max_dim_vals = self.object_manager.max_dim_vals
        self.max_dim_inds = self.object_manager.max_dim_inds
        self.task_flags = self.object_manager.task_flags
        self.is_place = self.object_manager.is_place
        self.is_pick = self.object_manager.is_pick

        # Randomize props for all envs
        self.object_manager.reset_object_shapes(torch.arange(self.num_envs, device=self.device))
        
        self.task_id = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.P_base = torch.zeros(self.num_envs, self.cfg.env.num_position, dtype=torch.float, device=self.device, requires_grad=False)
        
        # Initialize Grasp Pose Optimizer
        self.grasp_optimizer = GraspPoseOptimizer(self.cfg, self.device)
        self.distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.objct_z = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.object_moved = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)

        self.out_of_view_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.out_of_view_drift = torch.zeros(self.num_envs, 1, 3, device=self.device)
        
        self.physically_out_of_view = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        
        # Success Timer for Dense Reward
        self.success_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.is_success_state = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.is_failure_state = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)

        self.force_look_triggered = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.force_look_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        
        # Replay Buffers
        self.last_is_place = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_env_shape_type_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.last_object_dims = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.last_local_points = torch.zeros(self.num_envs, self.cfg.target.perception.num_sample_points, 3, dtype=torch.float, device=self.device)
        self.last_local_normals = torch.zeros(self.num_envs, self.cfg.target.perception.num_sample_points, 3, dtype=torch.float, device=self.device)
        self.last_obj_is_vertical = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_obj_is_too_long = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_object_z_offset = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.last_object_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.last_object_quat = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        self.last_root_states = torch.zeros(self.num_envs, 13, dtype=torch.float, device=self.device)

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

        self.near_target = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.far_target = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)

        self.nav_commands = torch.zeros(self.num_envs, self.cfg.commands.num_nav_commands, dtype=torch.float, device=self.device, requires_grad=False)

        self.nav_actions = torch.zeros(self.num_envs, self.cfg.env.num_nav_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_orig_nav_actions = torch.zeros(self.num_envs, self.cfg.env.num_nav_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_actions = torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device, requires_grad=False)

        self.nav_actions_buffer = torch.zeros(self.num_envs, self.cfg.env.history_len, self.cfg.env.num_nav_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.nav_commands_buffer = torch.zeros(self.num_envs, self.cfg.env.nav_history_len, self.cfg.commands.num_nav_commands, dtype=torch.float, device=self.device, requires_grad=False)
        self.nav_clip_min = torch.tensor([self.cfg.commands.ranges.limit_vx[0], self.cfg.commands.ranges.limit_vy[0], self.cfg.commands.ranges.limit_vyaw[0], self.cfg.commands.ranges.limit_pitch[0]], dtype=torch.float, device=self.device, requires_grad=False)
        self.nav_clip_max = torch.tensor([self.cfg.commands.ranges.limit_vx[1], self.cfg.commands.ranges.limit_vy[1], self.cfg.commands.ranges.limit_vyaw[1], self.cfg.commands.ranges.limit_pitch[1]], dtype=torch.float, device=self.device, requires_grad=False)
        self.obs_hist_buffer = torch.zeros(self.num_envs, self.cfg.env.history_len, self.cfg.env.num_props, dtype=torch.float, device=self.device, requires_grad=False)
        self.raw_obs_buffer = torch.zeros(self.num_envs, self.cfg.env.history_len + 10, self.cfg.env.num_props, dtype=torch.float, device=self.device, requires_grad=False)
        self.current_perceived_obs = torch.zeros(self.num_envs, self.cfg.env.num_props, dtype=torch.float, device=self.device, requires_grad=False)
        
        self.cmds_alpha = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        self.sigma_points_alpha = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        self.vis_weights = torch.zeros(self.num_envs, self.cfg.target.perception.num_sample_points, 1, device=self.device)
        self.is_valid = torch.ones(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)

        self.euler_rpy = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_lin_vel_pred = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.loco_obs_buf = torch.zeros(
                self.num_envs, self.cfg.loco.loco_obs_dim, device=self.device, dtype=torch.float)
        self.loco_obs_hist = torch.zeros(
                self.num_envs, self.cfg.loco.loco_history_len, self.cfg.loco.loco_obs_dim, device=self.device, dtype=torch.float)  

        self.loco_obs_noise_vec = self._get_loco_obs_noise_scale_vec()

        # Perception delay and frequency randomization
        # refresh_interval 1 = 50Hz, 2 = 25Hz (since dt=20ms)
        self.nav_refresh_interval = torch.ones(self.num_envs, device=self.device, dtype=torch.int64)
        self.nav_latency_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)

        self.dynamic_replay_prob = 0.0
        self.dynamic_feeding_prob = 0.0
        
        self._load_loco_policy()

    @property
    def x_axis_world(self):
        return self.object_manager.x_axis_world

    @property
    def y_axis_world(self):
        return self.object_manager.y_axis_world

    @property
    def z_axis_world(self):
        return self.object_manager.z_axis_world

    @property
    def axis_long_world(self):
        return self.object_manager.axis_long_world

    @property
    def p_head_world(self):
        return self.object_manager.p_head_world

    @property
    def p_tail_world(self):
        return self.object_manager.p_tail_world

    def _load_loco_policy(self):
        """ load loco policy, which is used to compute loco actions from nav actions
            the loco policy is a slr policy, which is trained with proprioception
        """
        self.loco_body = torch.jit.load(self.cfg.loco.loco_body_model_file)
        self.loco_body =  self.loco_body.to(self.device)

    def _compute_actions(self, nav_actions):
        """ nav_actions (loco_commands) -> loco_actions (self.commands)
            a hacky implementation
        """
        self.commands = self._smooth_nav_actions(nav_actions) # vx, vy, vyaw, pitch

        if self.cfg.commands.resample.force_look_upwards:
            # maintain looking upwards for a period of time
            self._force_look_upwards()

        self.commands = self.nav_actions
        self._compute_loco_observations()
        actor_obs = self.loco_obs_hist.view(self.num_envs, -1)
        loco_actions, self.base_lin_vel_pred = self.loco_body(actor_obs)
        return loco_actions
    
    def _smooth_nav_actions(self, nav_actions):
        """ Smooth the nav actions over time
        """
        # sample alpha from [0, 1], self.cmds_alpha is the smoothing factor, shape: (num_envs, 1)
        # in this way, we can reduce sudden changes in joints (alpha=0: no smoothing, but motors will respond immediately in reality)
        self.nav_actions = self.cmds_alpha * nav_actions + (1 - self.cmds_alpha) * self.nav_actions
        # self.nav_actions = nav_actions
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
        noise_vec[start:end] = 0.0 # self.commands
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
        self.props = torch.cat((
                self.base_ang_vel * self.obs_scales.ang_vel, # 3
                self.projected_gravity, # 3
                self.commands * self.commands_scale, # 4
                self.euler_rpy[:, 1:2] * self.obs_scales.pitch,  # dim 1
                self.reindex((self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos),
                self.reindex(self.dof_vel * self.obs_scales.dof_vel),
                self.last_dof_actions),dim=-1)
        
        if self.add_noise:
            self.props += (2 * torch.rand_like(self.props) - 1) * self.loco_obs_noise_vec
        
        self.loco_obs_hist = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([self.props] * self.loco_obs_hist.shape[1], dim=1),
            torch.cat([
                self.loco_obs_hist[:, 1:],
                self.props.unsqueeze(1)
            ], dim=1)
        )  

    def reset_idx(self, env_ids):
        """ origin: update terrain_cur(env_origin), cmd_curr, dofs, root_state(pos, vel), resample_cmds, fill extras
            new: remove cmd_curr, and some useless vars, update only when time out
        """
        
        if len(env_ids) == 0:
            return

        # --- Replay Logic ---
        current_success_ratio = torch.mean(self.is_success_state.float())
        replay_cfg = self.cfg.commands.resample
        
        # --- Adaptive Drift Curriculum ---
        if self.cfg.commands.enable_drift_curriculum:
            # Map success ratio [0, threshold] -> [min, max]
            threshold = self.cfg.commands.drift_curriculum_threshold
            progress = torch.clamp(current_success_ratio / threshold, 0.0, 1.0)
            
            # Linear interpolation
            d_scale_min, d_scale_max = self.cfg.commands.drift_scale_range
            m_drift_min, m_drift_max = self.cfg.commands.max_drift_range
            
            self.cfg.commands.drift_scale = d_scale_min + (d_scale_max - d_scale_min) * progress.item()
            self.cfg.commands.max_drift = m_drift_min + (m_drift_max - m_drift_min) * progress.item()

        # Conditions for replay:
        # 1. success_ratio > threshold
        # 2. was failure in last episode
        # 3. random chance
        threshold = getattr(replay_cfg, 'curriculum_threshold', 0.1)
        r_min, r_max = getattr(replay_cfg, 'replay_failed_prob_range', [0.2, 0.8])
        progress = torch.clamp(current_success_ratio / threshold, 0.0, 1.0)
        self.dynamic_replay_prob = r_max - (r_max - r_min) * torch.exp(-5.0 * progress).item()
        should_replay = (self.is_failure_state[env_ids]) & \
                        (torch.rand(len(env_ids), device=self.device) < self.dynamic_replay_prob)

        replay_env_ids = env_ids[should_replay]
        new_env_ids = env_ids[~should_replay]

        # 1. Handle New Scenarios
        if len(new_env_ids) > 0:
            self._reset_dofs(new_env_ids)
            self._reset_root_states(new_env_ids)
            self.object_manager.reset_object_shapes(new_env_ids)
            self._resample_commands(new_env_ids)
            self._save_scenario_state(new_env_ids)

        # 2. Handle Replay Scenarios
        if len(replay_env_ids) > 0:
            self._replay_scenario_state(replay_env_ids)

        self.env_start_pos[env_ids] = self.root_states[env_ids, :3].clone() # Store initial pos

        self.last_actions[env_ids] = 0.

        self.last_dof_vel[env_ids] = 0.
        self.last_root_vel[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.object_moved[env_ids] = False
        self.out_of_view_timer[env_ids] = 0.
        self.out_of_view_drift[env_ids] = 0.
        self.success_timer[env_ids] = 0.
        self.force_look_timer[env_ids] = 0.
        self.is_success_state[env_ids] = False
        self.is_failure_state[env_ids] = False
        self.force_look_triggered[env_ids] = False
        
        # Resample perception refresh interval
        refresh_range = self.cfg.commands.nav_refresh_steps_range
        self.nav_refresh_interval[env_ids] = torch.randint(refresh_range[0], refresh_range[1] + 1, (len(env_ids),), device=self.device)
        
        # Resample smoothing factor alpha from [0.1, 0.5] step 0.05
        cmds_alpha_range = self.cfg.commands.cmds_alpha_range
        cmds_alpha_step = self.cfg.commands.cmds_alpha_step
        alpha_values = torch.arange(cmds_alpha_range[0], cmds_alpha_range[1], cmds_alpha_step, device=self.device)
        alpha_indices = torch.randint(0, len(alpha_values), (len(env_ids),), device=self.device)
        self.cmds_alpha[env_ids] = alpha_values[alpha_indices].unsqueeze(1)
        
        # Resample sigma points alpha
        sp_alpha_range = self.cfg.target.perception.alpha_range
        self.sigma_points_alpha[env_ids] = torch_rand_float(sp_alpha_range[0], sp_alpha_range[1], (len(env_ids), 1), device=self.device)
        
        if hasattr(self, 'filter_reset_mask'):
            self.filter_reset_mask[env_ids] = True

        # Resample latency steps
        if self.cfg.commands.enable_delay:
            dt_ms = self.dt * 1000
            max_lat_step = int(self.cfg.commands.max_delay_time_ms // dt_ms)
            min_lat_step = int(self.cfg.commands.min_delay_time_ms // dt_ms)
            self.nav_latency_steps[env_ids] = torch.randint(min_lat_step, max_lat_step + 1, (len(env_ids),), device=self.device)
        else:
            self.nav_latency_steps[env_ids] = 0

        self.loco_obs_hist[env_ids, :, :] = 0.
        self.nav_commands_buffer[env_ids, :, :] = 0.
        self.nav_actions_buffer[env_ids, :, :] = 0.
        self.obs_hist_buffer[env_ids, :, :] = 0.
        self.raw_obs_buffer[env_ids, :, :] = 0.
        self.current_perceived_obs[env_ids, :] = 0.

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

        # Log Dynamic Replay Probability
        self.extras["dynamic_replay_prob"] = self.dynamic_replay_prob

        # Log Dynamic Feeding Probability
        self.extras["dynamic_feeding_prob"] = self.dynamic_feeding_prob


    def _save_scenario_state(self, env_ids):
        """ Save the current scenario state for potential future replay """
        self.last_is_place[env_ids] = self.is_place[env_ids]
        self.last_env_shape_type_indices[env_ids] = self.env_shape_type_indices[env_ids]
        self.last_object_dims[env_ids] = self.object_dims[env_ids]
        self.last_local_points[env_ids] = self.local_points[env_ids]
        self.last_local_normals[env_ids] = self.local_normals[env_ids]
        self.last_obj_is_vertical[env_ids] = self.obj_is_vertical[env_ids]
        self.last_obj_is_too_long[env_ids] = self.obj_is_too_long[env_ids]
        self.last_object_z_offset[env_ids] = self.object_z_offset[env_ids]
        self.last_object_pos[env_ids] = self.object_pos[env_ids]
        self.last_object_quat[env_ids] = self.object_quat[env_ids]
        self.last_root_states[env_ids] = self.root_states[env_ids]

    def _replay_scenario_state(self, env_ids):
        """ Restore the scenario state from the last saved state """
        self._reset_dofs(env_ids)
        # Restore robot state
        self.root_states[env_ids] = self.last_root_states[env_ids]
        
        # Restore object properties
        self.is_place[env_ids] = self.last_is_place[env_ids]
        self.env_shape_type_indices[env_ids] = self.last_env_shape_type_indices[env_ids]
        self.object_dims[env_ids] = self.last_object_dims[env_ids]
        self.local_points[env_ids] = self.last_local_points[env_ids]
        self.local_normals[env_ids] = self.last_local_normals[env_ids]
        self.obj_is_vertical[env_ids] = self.last_obj_is_vertical[env_ids]
        self.obj_is_too_long[env_ids] = self.obj_is_too_long[env_ids]
        self.object_z_offset[env_ids] = self.last_object_z_offset[env_ids]
        self.object_pos[env_ids] = self.last_object_pos[env_ids]
        self.object_quat[env_ids] = self.last_object_quat[env_ids]
        
        # Update task flags and other derived quantities
        self.is_pick[env_ids] = ~self.is_place[env_ids]
        self.task_flags[env_ids] = self.is_place[env_ids].float().unsqueeze(-1)
        
        # Update tracker
        self.tracker.object_dims[env_ids] = self.object_dims[env_ids]
        self.tracker.local_points[env_ids] = self.local_points[env_ids]
        self.tracker.local_normals[env_ids] = self.local_normals[env_ids]
        
        self.max_dim_vals[env_ids], self.max_dim_inds[env_ids] = torch.max(self.object_dims[env_ids], dim=1)

        # Re-sync physics state for root
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
    def reindex(self,tensor):
        """ sim2real purpose
        """
        return tensor[:,[3,4,5,0,1,2,9,10,11,6,7,8]]

    def step(self, nav_actions):
        """ origin: loco policy (dim12) -> act2tq
            new: nav policy (dim 3): loco_cmds -> act (dim 12)
        """
        # nav_actions: vx, vy, vyaw, pitch, gripper_state
        nav_actions = torch.clip(nav_actions, min=-3.0, max=3.0)
        self.orig_nav_actions = nav_actions.to(self.device)
        nav_actions_after_clip = torch.clip(self.orig_nav_actions, min=self.nav_clip_min, max=self.nav_clip_max)
        self.nav_actions_after_clip = nav_actions_after_clip.clone()
        actions = self._compute_actions(self.nav_actions_after_clip) # just smooth actions here
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
            # if self.device == 'cpu':
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
            self.last_fpv_frame = self._draw_debug_vis()
        else:
            self.last_fpv_frame = None

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        # --- Success/Failure Check ---
        self._check_task_state()

        self.reset_buf |= self.time_out_buf
        self.reset_buf |= self.reach_goal

    def _check_task_state(self):
        """ Check if the current gripper pose is at the optimal grasp pose
        Conditions:
            1. Distance to Optimal Grasp Pose < Threshold
            2. Orientation Alignment < Threshold
        """
        # 1. Distance Check (Close to Optimal Grasp Pose)
        self.near_target = torch.logical_and(self.d_goal_err_x < 0.03, self.d_goal_err_y < 0.03)
        self.far_target = torch.logical_or(self.d_goal_err_x >= 0.2, self.d_goal_err_y > 0.05)

        # 2. Orientation Alignment
        # is_aligned = self.yaw_err < 0.1 # ~5.7 degrees tolerance
        is_at_target = self.is_pick * torch.logical_and(self.d_goal_err_x < 0.03, self.d_goal_err_y < 0.03) + \
                         self.is_place * torch.logical_and(self.d_goal_err_x < 0.05, self.d_goal_err_y < 0.05)
        
        is_aligned = self.is_pick * (self.yaw_err < 0.1) + \
                        self.is_place * ( (self.yaw_err < 0.1) & (self.pitch_err < 0.15) )
        self.is_success_state = is_at_target & is_aligned
        
        is_far_from_target = self.is_pick * torch.logical_or(self.d_goal_err_x >= 0.1, self.d_goal_err_y >= 0.1) + \
                                self.is_place * torch.logical_or(self.d_goal_err_x >= 0.2, self.d_goal_err_y >= 0.1)
        is_far_from_aligned = self.is_pick * self.obj_is_too_long & (self.yaw_err >= 0.2) + \
                                self.is_place * ( (self.yaw_err >= 0.2) | (self.pitch_err >= 0.2) )
        self.is_failure_state = self.time_out_buf & (is_far_from_target | is_far_from_aligned)

        # Update timer
        self.success_timer[self.is_success_state] += self.dt
        self.success_timer[~self.is_success_state] = 0.0
        # Terminate only after holding for 1 seconds
        self.reach_goal = self.success_timer > self.cfg.commands.hold_time_s

    def _compute_device_pose_in_world(self):
        """ compute camera pose in world frame
        """
        # Compute Gripper Position in World Frame
        self.gripper_world = self.root_states[:, :3] + quat_apply(self.base_quat, self.gripper_offset)

        # compute camera pose in world frame
        self.camera_world = quat_apply(self.base_quat, self.camera_sensor.T) + self.root_states[:, 0:3]
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
        # valid_time = self.episode_length_buf > self.cfg.commands.resample.min_steps
        valid_time = self.episode_length_buf % self.cfg.commands.resample.resample_interval_steps == 0
        # to_move = ((~self.object_moved) & valid_time)
        to_move = valid_time
        env_ids_resample = to_move.nonzero(as_tuple=False).flatten()

        if len(env_ids_resample) > 0:
            # Check config for active success feeding
            use_success_feeding = self.cfg.commands.resample.enable_success_feeding
        
            if use_success_feeding:
                # Only feed a subset of environments to encourage exploration
                # Probability of feeding: decreases as success ratio increases
                current_success_ratio = torch.mean(self.is_success_state.float())
                
                resample_cfg = self.cfg.commands.resample
                threshold = getattr(resample_cfg, 'curriculum_threshold', 0.1)
                f_min, f_max = getattr(resample_cfg, 'feeding_prob_range', [0.2, 0.5])
                progress = torch.clamp(current_success_ratio / threshold, 0.0, 1.0)
                self.dynamic_feeding_prob = f_min + (f_max - f_min) * torch.exp(-5.0 * progress).item()
                
                should_feed = torch.rand(len(env_ids_resample), device=self.device) < self.dynamic_feeding_prob
                # invalid env ids
                # should_feed = should_feed | (~self.is_valid[env_ids_resample])
                ids_to_feed = env_ids_resample[should_feed]
                if len(ids_to_feed) > 0:
                    self.object_manager.resample_object_pose_near_success(ids_to_feed, self.gripper_world[ids_to_feed], self.base_quat[ids_to_feed], self.dist_lat_range)
                    self._save_scenario_state(ids_to_feed)
                
            else:
                self._resample_commands(env_ids_resample)
                self.env_start_pos[env_ids_resample] = self.root_states[env_ids_resample, :3].clone()
            
            self.object_moved[env_ids_resample] = True

    def _compute_optimal_hint_pose(self):
        """
        Compute the optimal grasp pose and hint pose based on object shape.
        Delegates logic to GraspPoseOptimizer.
        """
        (self.optimal_grasp_pos, 
         self.optimal_grasp_quat, 
         self.optimal_approach_dir, 
         self.hint_pos, 
         self.hint_quat) = self.grasp_optimizer.compute_optimal_hint_pose(
            robot_base_pos=self.root_states[:, :3],
            gripper_pos=self.gripper_world,
            object_pos=self.object_pos,
            object_dims=self.object_dims,
            p_head_world=self.p_head_world,
            p_tail_world=self.p_tail_world,
            axis_long_world=self.axis_long_world,
            env_start_pos=self.env_start_pos,
            obj_is_too_long=self.obj_is_too_long,
            is_place=self.is_place,
            is_pick=self.is_pick
        )
        
    def _compute_optimal_pose_error(self):
        """ Compute optimal pose errors from current gripper pose to optimal grasp pose
        1. Position Error (Weighted Squared)
        2. Orientation Error (Weighted Squared)
        """
        # Position Error
        w_x = self.cfg.rewards.opt_pos_weights.x
        w_y = self.cfg.rewards.opt_pos_weights.y
        w_z = self.cfg.rewards.opt_pos_weights.z
        self.d_goal_err_x = torch.abs(self.gripper_world[:, 0] - self.optimal_grasp_pos[:, 0])
        self.d_goal_err_y = torch.abs(self.gripper_world[:, 1] - self.optimal_grasp_pos[:, 1])
        self.d_goal_err_z = torch.abs(self.gripper_world[:, 2] - self.optimal_grasp_pos[:, 2])
        self.pos_error_sq = w_x * torch.square(self.d_goal_err_x) + w_y * torch.square(self.d_goal_err_y) + w_z * torch.square(self.d_goal_err_z)
        self.pos_error_sq_place = 2 * torch.square(self.d_goal_err_x) + 0.2 * torch.square(self.d_goal_err_y) + 0.1 * torch.square(self.d_goal_err_z)
        
        # Orientation Error
        roll_r, pitch_r, yaw_r = get_euler_xyz(self.base_quat)
        roll_t, pitch_t, yaw_t = get_euler_xyz(self.optimal_grasp_quat)
        # pitch_t_clipped = torch.clamp(pitch_t, min=self.nav_clip_min[-1], max=self.nav_clip_max[-1])
        pitch_t_clipped = torch.clamp(wrap_to_pi(pitch_t), min=-0.25, max=0.40)  # Clip between -24 to 24 degrees
        pitch_r_warped = wrap_to_pi(pitch_r)

        self.roll_err = torch.abs(wrap_to_pi(roll_r - roll_t))
        self.pitch_err = torch.abs(pitch_r_warped - pitch_t_clipped)
        self.yaw_err = torch.abs(wrap_to_pi(yaw_r - yaw_t))
        
        w_yaw = self.cfg.rewards.opt_rot_weights.yaw
        w_pitch = self.cfg.rewards.opt_rot_weights.pitch
        w_roll = self.cfg.rewards.opt_rot_weights.roll
        self.rot_error_sq_far = w_yaw * torch.square(self.yaw_err) # far from target only cares about yaw
        self.rot_error_sq_near = 0.7 * torch.square(self.yaw_err) + 0.3 * torch.square(self.pitch_err) + w_roll * torch.square(self.roll_err) # near target cares about all axes

    def _post_physics_step_callback(self):
        """ origin: resample_cmds[env_ids] at resampling_time
            new: update nav_commands every step
        """
        if self.cfg.commands.resample.adjust_obj_pose:
            self.resample_object_pose_on_the_way()
        
        # update distance and reach_goal
        self.distance = torch.norm(self.root_states[:, :2] - self.object_pos[:, :2], dim=1)
        # Robot forward vector in world frame
        self.forward_world = quat_apply(self.base_quat, self.forward_vec)

        # get self.camera_transform_world('R': self.R_world_to_cam, 'T': self.camera_world), self.gripper_world
        self._compute_device_pose_in_world()

        # get optimal grasp pose
        self._compute_optimal_hint_pose() # Nontrivial

        # get self.sigma_points_3d, self.sigma_points_base, self.sigma_points_2d, self.is_valid for perception in obs
        self._get_nav_commands() # Nontrivial

        # get self.pos_error_sq, self.rot_error_sq
        self._compute_optimal_pose_error() # Nontrivial

        self.nav_actions_buffer = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([self.nav_actions] * self.nav_actions_buffer.shape[1], dim=1),
            torch.cat([
                self.nav_actions_buffer[:, 1:],
                self.nav_actions.unsqueeze(1)
            ], dim=1)
        ) 

        self.timer = (self.episode_length_buf / self.max_episode_length).unsqueeze(-1)

    def _force_look_upwards(self):
        """ Forcing the robot to look upwards in place task, even if it loses sight of the target.
        """
        # Near place condition
        near_place = self.near_target & self.is_place
        near_place_env_ids = near_place.nonzero(as_tuple=False).flatten()

        # Filter environments that have not triggered yet
        untriggered_envs = near_place_env_ids[~self.force_look_triggered[near_place_env_ids]]
        should_pitch = torch.rand(len(untriggered_envs), device=self.device) < self.cfg.commands.resample.force_lookup_prob
        env_ids = untriggered_envs[should_pitch]

        # Set pitch and mark as triggered
        if len(env_ids) > 0:
            roll_t, pitch_t, yaw_t = get_euler_xyz(self.optimal_grasp_quat[env_ids])
            self.force_look_triggered[env_ids] = True
            self.force_look_timer[env_ids] = self.cfg.commands.resample.look_up_duration

        # Maintain look-up action for triggered environments
        active_envs = self.force_look_triggered.nonzero(as_tuple=False).flatten()
        if len(active_envs) > 0:
            roll_t, pitch_t, yaw_t = get_euler_xyz(self.optimal_grasp_quat[active_envs])
            self.nav_actions[active_envs, -1] = torch.clamp(wrap_to_pi(pitch_t), min=self.nav_clip_min[-1], max=self.nav_clip_max[-1])  # (vx, vy, vz, pitch)

        # Decrease timer for all environments
        self.force_look_timer -= self.dt
        self.force_look_timer = torch.clamp(self.force_look_timer, min=0.0)

        # Reset trigger for environments where timer has expired
        expired_envs = (self.force_look_timer == 0.0)
        self.force_look_triggered[expired_envs] = False

    def _resample_commands(self, env_ids):
        """ Set the object pose in the visualable zone and update related world frame quantities
        """
        if len(env_ids) == 0:
            return
        
        # 1. Resample Object Position & Orientation via Manager
        self.object_manager.resample_object_positions(
            env_ids, 
            cam_transform=self.camera_transform_dict, 
            root_states=self.root_states
        )
        self.object_manager.resample_object_orientations(env_ids)

        # Get object axis and end points in world frame
        self.object_manager.compute_object_axis_end_points_in_world()
        
    def _run_perception_tracker(self):
        """ Runs the tracker to get raw Sigma Points and validity. """
        # Prepare structured noise
        noise_params = None
        if self.add_noise:
            noise_scales = self.cfg.noise.noise_scales
            noise_params = {
                'pos_3d': noise_scales.nav_pos_3d,
                'scale_3d': noise_scales.nav_scale_3d,
                'rot_3d': noise_scales.nav_rot_3d,
                'pos_2d': noise_scales.nav_pos_2d,
                'scale_2d': noise_scales.nav_scale_2d,
                'rot_2d': noise_scales.nav_rot_2d,
            }

        # Prepare pre-PCA noise
        pre_pca_noise_params = None
        if self.cfg.target.perception.add_pre_pca_noise:
            pre_pca_noise_params = self.cfg.target.perception.pre_pca_noise

        # Run Tracker
        return self.tracker.compute_features(
            object_pos=self.object_pos,
            object_quat=self.object_quat,
            camera_params=self.camera_params_dict,
            camera_transform=self.camera_transform_world,
            use_geometric_weight=self.use_geometric_weight,
            noise_params=noise_params,
            pre_pca_noise_params=pre_pca_noise_params,
            noise_active_mask=self.is_place,
            debug_timer=self.debug_timer,
            debug_info=self.debug_info,
            alpha=self.sigma_points_alpha
        )

    def _estimate_object_state(self, sigma_points_3d, is_valid):
        """ Handles simulated Kalman Filter logic (prediction & filtering) """
        # --- Handle Invalid / Out of View (Simulation of Kalman Filter Prediction) ---
        if not hasattr(self, 'last_valid_sigma_points_world'):
            self.last_valid_sigma_points_world = sigma_points_3d.clone()

        # Handle Resets
        reset_mask = self.episode_length_buf <= 1
        if reset_mask.any():
            self.last_valid_sigma_points_world[reset_mask] = sigma_points_3d[reset_mask]

        # Update Last Valid
        if is_valid.any():
            self.last_valid_sigma_points_world[is_valid] = sigma_points_3d[is_valid]

        # Prediction: Collapse to last valid center if invalid
        valid_mask = is_valid.view(self.num_envs, 1, 1)
        last_valid_center = self.last_valid_sigma_points_world[:, 0:1, :]
        prediction_collapsed = last_valid_center.expand(-1, sigma_points_3d.shape[1], -1)
        sigma_points_filled = torch.where(valid_mask, sigma_points_3d, prediction_collapsed)

        # --- Filter Sigma Points (EMA) ---
        if not hasattr(self, 'sigma_points_filtered'):
            self.sigma_points_filtered = sigma_points_filled.clone()
            self.filter_reset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Reset filters
        if self.filter_reset_mask.any():
            self.sigma_points_filtered[self.filter_reset_mask] = sigma_points_filled[self.filter_reset_mask]
            self.filter_reset_mask[:] = False
            
        # Apply EMA
        alphas = torch.where(self.is_place, 0.4, 1.0).view(-1, 1, 1)
        self.sigma_points_filtered = torch.lerp(self.sigma_points_filtered, sigma_points_filled, alphas)
        return self.sigma_points_filtered.clone()

    def _apply_sensor_artifacts(self, sigma_points_camera, is_valid):
        """ Handles sensor imperfections: frame drops, drift, invalid masking """
        sigma_points_obs_unmasked = sigma_points_camera.clone()
        invalid_sigma = torch.ones_like(sigma_points_obs_unmasked) * -1.0
        
        valid_mask = is_valid.view(self.num_envs, 1, 1)

        if self.cfg.commands.enable_invalid_cmds:
            if self.cfg.commands.enable_out_of_view_drift:
                # Reset drift for those in view
                self.out_of_view_drift[is_valid] = 0.0
                
                # Add drift step for those out of view
                drift_step = torch.randn_like(self.out_of_view_drift[~is_valid]) * self.cfg.commands.drift_scale
                drift_step[:, :, 2] += self.cfg.commands.drift_scale # Drift forward (+Z)

                self.out_of_view_drift[~is_valid] += drift_step
                self.out_of_view_drift = torch.clamp(self.out_of_view_drift, 
                                                    min=-self.cfg.commands.max_drift, 
                                                    max=self.cfg.commands.max_drift)
                sigma_points_obs_unmasked += self.out_of_view_drift
                return sigma_points_obs_unmasked
            else:
                return torch.where(valid_mask, sigma_points_obs_unmasked, invalid_sigma)
        else:
            return sigma_points_obs_unmasked

    def _update_nav_buffers(self, nav_commands):
        """ Updates the navigation command history buffer """
        env_ids = (self.episode_length_buf % 10 == 0).nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.nav_commands_buffer[env_ids] = torch.where(
                (self.episode_length_buf[env_ids] <= 1)[:, None, None],
                torch.stack([nav_commands[env_ids]] * self.nav_commands_buffer[env_ids].shape[1], dim=1),
                torch.cat([
                    self.nav_commands_buffer[env_ids, 1:],
                    nav_commands[env_ids].unsqueeze(1)
                ], dim=1)
            )

    def _get_nav_commands(self):
        """ Compute perception features (Sigma Points) and update nav_commands.
        """
        # 1. Perception (Raw Features)
        sigma_points_2d, _, _, _, weights, _, sigma_points_3d, is_valid = self._run_perception_tracker()
        self.vis_weights[:] = weights

        # 2. State Estimation (KF & Filtering)
        sigma_points_3d_est = self._estimate_object_state(sigma_points_3d, is_valid)

        # 3. Simulate Frame Drops (affects validity for artifacts)
        if self.cfg.commands.frame_drop_prob > 0:
            frame_drop_mask = torch.rand(self.num_envs, device=self.device) < self.cfg.commands.frame_drop_prob
            is_valid &= ~frame_drop_mask

        # 4. Transform to Robot Frame (Base -> Camera)
        self.sigma_points_base, self.sigma_points_camera, _ = self.tracker.transform_sigma_points_to_robot(
            sigma_points_3d_est, self.root_states[:, :3], self.root_states[:, 3:7],
            self.camera_sensor.T, self.camera_sensor.R, 
            self.camera_params_dict
        )

        # 5. Apply Sensor Artifacts (Drift / Invalid Masking)
        self.sigma_points_obs = self._apply_sensor_artifacts(self.sigma_points_camera, is_valid)

        # 6. Update Buffers & State
        self.nav_commands = self.sigma_points_obs[:, :self.cfg.env.num_nav_commands//3].reshape(self.num_envs, -1)
        self._update_nav_buffers(self.nav_commands)

        self.all_valid_sigma_points = self.sigma_points_camera.clone().reshape(self.num_envs, -1) # Use unmasked for privileged
        self.sigma_points_3d = sigma_points_3d
        self.sigma_points_2d = sigma_points_2d
        self.is_valid = is_valid

    def _draw_debug_vis(self, sigma_points_3d=None, sigma_points_2d=None, force_fpv=False):
        """ Draw Sigma Points in 3D and 2D """
        fpv_frame = None
        if sigma_points_3d is None:
            if hasattr(self, 'sigma_points_3d'):
                sigma_points_3d = self.sigma_points_3d
            else:
                return fpv_frame
        
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
                env_idx=0,
                weights=self.vis_weights[0]
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
            if self.common_step_counter % 10 == 0 or force_fpv:
                points_3d_dict = {}
                if self.cfg.camera_sensor.vis_target_points_in_image:
                    points_3d_dict["target"] = points_world_sample
                if self.cfg.camera_sensor.vis_sigma_3d_in_image:
                    points_3d_dict["sigma_3d"] = sigma_points_3d[0][0:self.cfg.commands.num_nav_commands//3]
                
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

                fpv_frame = self.vis_utils.draw_2d_image(
                    points_3d_dict=points_3d_dict, 
                    camera_sensor=self.camera_sensor, 
                    env_idx=0, 
                    filename=f"debug_cam_{self.common_step_counter}.png",
                    save_images=self.save_debug_images,
                    points_2d_dict=points_2d_dict,
                    lines_3d_dict=lines_dict
                )
        return fpv_frame
    
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
        end = 0

        start = end
        end = start + 3 # base_lin_vel_pred
        noise_vec[start:end] = 0. # usually no direct noise for estimated velocity

        start = end
        end = start + self.base_ang_vel.shape[1]
        noise_vec[start:end] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel

        start = end
        end = start + self.projected_gravity.shape[1]
        noise_vec[start:end] = noise_scales.gravity * noise_level * 1.0

        start = end
        end = start + self.nav_commands.shape[1]
        noise_vec[start:end] = 0. * noise_level # sigma points (noise already added in _get_nav_commands)

        start = end
        end = start + self.task_flags.shape[1]
        noise_vec[start:end] = 0. # task_flags

        start = end
        end = start + self.nav_actions.shape[1]
        noise_vec[start:end] = 0. # self.nav_actions (get from rl policy, no noise)

        return noise_vec

    def compute_observations(self):
        """ Computes observations for updating nav agent with simulated latency and jitter
        """
        
        # [Contrastive Learning] Update Task ID
        # 0: Pick Long, 1: Pick Short, 2: Place
        self.task_id[:] = 0.0
        self.task_id[self.is_pick & (~self.obj_is_too_long)] = 1.0
        self.task_id[self.is_place] = 2.0

        obs_now = torch.cat([
            self.base_lin_vel_pred * self.obs_scales.lin_vel, # 3
            self.base_ang_vel * self.obs_scales.ang_vel, # 3
            self.projected_gravity, # 3
            self.nav_commands, # 15/21/9 (Ground Truth Vision)
            self.task_flags, # 1
            self.nav_actions, # 4
            ], dim=-1)

        # raw_obs_buffer: obs without delay/jitter
        self.raw_obs_buffer = torch.cat([
            self.raw_obs_buffer[:, 1:], 
            obs_now.unsqueeze(1)
            ], dim=1)

        # Simulate Frame Refresh Jitter and Latency
        if self.cfg.commands.enable_delay:
            # Frequency Jitter: Only "refresh" the sensor at randomized intervals
            refresh_mask = (self.episode_length_buf % self.nav_refresh_interval == 0)
            
            # Latency: Pick a frame from the past
            env_indices = torch.arange(self.num_envs, device=self.device)
            delayed_indices = self.raw_obs_buffer.shape[1] - 1 - self.nav_latency_steps
            delayed_obs = self.raw_obs_buffer[env_indices, delayed_indices, :]
            
            # Apply Frame-Drop/Refresh Jitter: 
            # If no refresh, keep the last perceived observation (frame frozen)
            self.current_perceived_obs[refresh_mask] = delayed_obs[refresh_mask]
        else:
            self.current_perceived_obs = obs_now

        self.obs_hist_buffer = torch.cat([
                self.obs_hist_buffer[:, 1:],
                self.current_perceived_obs.unsqueeze(1)
            ], dim=1)
        
        # return obs: priv + nav_cmd_hist + obs_hist
        self.obs_buf = torch.cat([
                                # ------- privileged information start -------
                                self.task_id, # dim 1
                                self.all_valid_sigma_points, # dim 21
                                # ------- privileged information end -------
                                self.nav_commands_buffer.view(self.num_envs, -1), # dim 21*5, TCN handling 
                                self.obs_hist_buffer.view(self.num_envs, -1)], dim=-1) # dim 35*5

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
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * (~self.force_look_triggered)
    
    def _reward_orientation_y(self):
        return torch.square(self.projected_gravity[:, 1]) * (~self.force_look_triggered)
    
    def _reward_nav_action_rate(self):
        return torch.square(torch.norm(self.orig_nav_actions - self.last_orig_nav_actions, dim=-1)) * (~self.force_look_triggered)

    def _reward_nav_action_limit(self):
        return torch.square(torch.norm(self.nav_actions_after_clip - self.orig_nav_actions, dim=-1)) * (~self.force_look_triggered)

    def _reward_missing_sigma_points(self):
        """ Reward for missing sigma points in the view
        """
        all_allow_visible = not self.cfg.commands.enable_invalid_cmds
        near_goal = self.distance < 1.0
        if all_allow_visible:
            return torch.zeros_like(self.is_valid).float()
        pick_short_success = self.is_pick * (~self.obj_is_too_long) * near_goal
        place_near_success = (self.is_place) * near_goal
        return (~self.is_valid).float() * (~pick_short_success) * (~place_near_success)

    def _reward_successful_grasp(self):
        """
        Reward for being in the success state (reaching object center with correct alignment).
        Given continuously while the robot maintains the state.
        """
        r_rot = torch.exp(-self.rot_error_sq_near / self.cfg.rewards.rot_track_sigma)
        r_pos_short_pick = torch.exp(-self.pos_error_sq / self.cfg.rewards.pos_track_sigma)
        r_pos_long_pick = torch.exp(-self.pos_error_sq / self.cfg.rewards.pos_track_sigma)
        r_pos_place = torch.exp(-self.pos_error_sq_place / self.cfg.rewards.soft_sigma)

        lin_vel_sq = torch.sum(torch.square(self.base_lin_vel), dim=1)
        ang_vel_sq = torch.sum(torch.square(self.base_ang_vel), dim=1)
        r_vel = torch.exp(-(lin_vel_sq + ang_vel_sq) / self.cfg.rewards.vel_track_sigma)
        _stand_still = torch.logical_and(self.base_lin_vel[:, 0] < 0.1, self.base_lin_vel[:, 0] > 0.0)  # robot should slow down when approaching the target
        _stand_still = torch.logical_and(_stand_still, self.base_ang_vel[:, 2].abs() < 0.1)  # reduce angular velocity
        _no_rotation = self.base_ang_vel[:, 2].abs() < 0.1  # reduce angular velocity
        _rew_short_pick = self.is_pick * (~self.obj_is_too_long) * (r_rot) * (1.0 + self.cfg.rewards.weight_track_pick_short_pos * r_pos_short_pick)
        _rew_long_pick = self.is_pick * (self.obj_is_too_long) * (r_rot) * (1.0 + 1.0 * self.cfg.rewards.weight_track_pick_long_pos * r_pos_long_pick)
        _rew_pick = _rew_short_pick * (1.0 + _stand_still) + _rew_long_pick
        _rew_place = self.is_place * (1.0 + r_rot) * (1.0 + self.cfg.rewards.weight_track_place_pos * r_pos_place) * _no_rotation
        return self.is_success_state.float() * (_rew_pick + _rew_place) * r_vel

    def _reward_backup(self):
        """ Reward for backing up when too close to the object
        """
        too_close = self.distance < 0.5
        backing_up = self.base_lin_vel[:, 0] < -0.1
        long_obj_pick = self.is_pick & self.obj_is_too_long # only disable when picking long objects
        return too_close.float() * backing_up.float() * (~long_obj_pick).float()
    
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
        d_path_diff_2d = gripper_pos_2d - p_closest_2d
        
        _slow_approach = torch.logical_and(self.base_lin_vel[:, 0] < 0.2, self.base_lin_vel[:, 0] > 0.0)  # robot should slow down when approaching the target
        _no_rotation = self.base_ang_vel[:, 2].abs() < 0.1  # reduce angular velocity
        # Rewards
        # 1. Path Following Reward (Pulls to Hint if t<0, then keeps on line)
        d_path_sq = torch.sum(torch.square(d_path_diff_2d), dim=-1)
        r_path = torch.exp(-d_path_sq / self.cfg.rewards.pos_track_sigma)
        
        # 2. Orientation Reward
        r_rot = (~self.near_target).float() * torch.exp(-self.rot_error_sq_far / self.cfg.rewards.rot_track_sigma) + \
                1 * (self.near_target).float() * torch.exp(-self.rot_error_sq_near / self.cfg.rewards.rot_track_sigma)
        
        r_pos_short_pick = torch.exp(-self.pos_error_sq / self.cfg.rewards.pos_track_sigma)
        r_pos_long_pick = torch.exp(-self.pos_error_sq / self.cfg.rewards.pos_track_sigma)
        r_pos_place = torch.exp(-self.pos_error_sq_place / self.cfg.rewards.soft_sigma)
        
        # 3. Goal Reaching Reward (Pulls along line to Optimal)
        _rew_short_pick = self.is_pick * (~self.obj_is_too_long) * (1 + 1.0 * r_path) * (r_rot) * (1.0 + 1.0 * self.cfg.rewards.weight_track_pick_short_pos * r_pos_short_pick)
        _rew_long_pick = self.is_pick * (self.obj_is_too_long) * (1 + 2.0 * r_path) * (r_rot) * (1.0 + 1.0 * self.cfg.rewards.weight_track_pick_long_pos * r_pos_long_pick)
        _rew_pick = _rew_short_pick * _slow_approach + _rew_long_pick
        _rew_place = self.is_place * r_path * (1.0 + r_rot) * (1.0 + self.cfg.rewards.weight_track_place_pos * r_pos_place) * _no_rotation
        
        return (_rew_pick + _rew_place)