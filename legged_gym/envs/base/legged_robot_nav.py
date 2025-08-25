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
from legged_gym.utils.camera_sensor import CameraSensor

class LeggedRobotNav(LeggedRobot):
    cfg : Go2NavFlatCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        self.debug_viz = self.cfg.debug_viz
        self.sigma = 0.1
        self.camera_sensor = CameraSensor(
            batch_size=self.num_envs, 
            cfg=self.cfg.camera_sensor,
            device=self.device,
            )

    def _init_buffers(self):
        """ inherit loco vars: self.commands[vx, vy, vyaw, pitch], self.actons[joint_pos]
            add nav vars: self.nav_commands[theta, rho], self.nav_actions[vx, vy, vyaw] = self.commands[:, :3]
            update vars: self.obs_buf -> nav_polciy
        """
        super()._init_buffers()
        
        self.position_targets = torch.zeros(self.num_envs, self.cfg.env.num_position, dtype=torch.float, device=self.device, requires_grad=False) # (x, y, z), align quat_rotate_inverse
        self.goal_base = torch.zeros(self.num_envs, self.cfg.env.num_position, dtype=torch.float, device=self.device, requires_grad=False)
        self.distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.objct_z = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        self.nav_commands = torch.zeros(self.num_envs, self.cfg.commands.num_nav_commands, dtype=torch.float, device=self.device, requires_grad=False)
        self.nav_actions = torch.zeros(self.num_envs, self.cfg.env.num_nav_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.nav_actions_before_clip = torch.zeros(self.num_envs, self.cfg.env.num_nav_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_nav_actions = torch.zeros(self.num_envs, self.cfg.env.num_nav_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_actions = torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device, requires_grad=False)

        self.nav_actions_buffer = torch.zeros(self.num_envs, self.cfg.env.history_len, self.cfg.env.num_nav_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.nav_commands_buffer = torch.zeros(self.num_envs, self.cfg.env.history_len, self.cfg.commands.num_nav_commands, dtype=torch.float, device=self.device, requires_grad=False)
        self.obs_hist_buffer = torch.zeros(self.num_envs, self.cfg.env.history_len, self.cfg.env.num_props, dtype=torch.float, device=self.device, requires_grad=False)
        self.nav_clip_min = torch.tensor([self.cfg.commands.ranges.limit_vx[0], self.cfg.commands.ranges.limit_vy[0], self.cfg.commands.ranges.limit_vyaw[0], self.cfg.commands.ranges.limit_pitch[0]], dtype=torch.float, device=self.device, requires_grad=False)
        self.nav_clip_max = torch.tensor([self.cfg.commands.ranges.limit_vx[1], self.cfg.commands.ranges.limit_vy[1], self.cfg.commands.ranges.limit_vyaw[1], self.cfg.commands.ranges.limit_pitch[1]], dtype=torch.float, device=self.device, requires_grad=False)

        self.base_euler = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.loco_obs_buf = torch.zeros(
                self.num_envs, 47, device=self.device, dtype=torch.float)
        self.loco_obs_hist = torch.zeros(
                self.num_envs, 10, 47, device=self.device, dtype=torch.float)  


        self._load_loco_policy()

    def _load_loco_policy(self):
        """ load loco policy, which is used to compute loco actions from nav actions
            the loco policy is a slr policy, which is trained with proprioception
        """
        
        # self.loco_body = torch.jit.load('controller/np3o/body_latest.jit')
        # self.loco_encoder_vel = torch.jit.load('controller/np3o/encoder_vel.jit')
        # self.loco_encoder_vel =  self.loco_encoder_vel.to(self.device)

        self.loco_body = torch.jit.load('controller/pitch/policy_pitch.jit')
        self.loco_body =  self.loco_body.to(self.device)

    def _compute_actions(self, nav_actions):
        """ nav_actions (loco_cmds) -> loco_actions (self.commands)
            a hacky implementation
        """
        self.commands = nav_actions # vx, vy, vyaw, pitch
        self._compute_loco_observations()
        actor_obs = self.loco_obs_hist.view(self.num_envs, -1)
        loco_actions, loco_lin_vel = self.loco_body(actor_obs)
            
        return loco_actions
    
    def _compute_loco_observations(self):
        """ It is only used for computing loco actions, NOT for updating rl agent.
        """
        props = torch.cat((
                self.base_ang_vel * self.obs_scales.ang_vel, # 3
                self.projected_gravity, # 3
                self.commands * self.commands_scale,
                self.base_euler[:, 1:2] * self.obs_scales.pitch,  # dim 1
                self.reindex((self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos),
                self.reindex(self.dof_vel * self.obs_scales.dof_vel),
                self.last_dof_actions),dim=-1)
        
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
        self.loco_obs_hist[env_ids, :, :] = 0.
        self.nav_commands_buffer[env_ids, :, :] = 0.
        self.nav_actions_buffer[env_ids, :, :] = 0.
        self.obs_hist_buffer[env_ids, :, :] = 0.

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
        roll, pitch, yaw = get_euler_xyz(self.base_quat)
        self.base_euler[:, 0] = wrap_to_pi(roll)
        self.base_euler[:, 1] = wrap_to_pi(pitch)
        self.base_euler[:, 2] = wrap_to_pi(yaw)

        self._post_physics_step_callback() # [new] update nav_commands

        # compute observations, rewards, resets, ...
        self.check_termination() # time out or collision
        self.compute_reward() # [new] add goal-reaching reward
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids) # [new] terrain_cur(env_origin), reset robot, resample_cmds
        self.update_first_image()
        self.compute_observations() # [new] update nav_commands for nav policy

        self.last_actions[:] = self.actions[:]
        self.last_nav_actions[:] = self.nav_actions[:]
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
        self.reset_buf |= (self.out_of_view * self.reach_goal)

    def _post_physics_step_callback(self):
        """ origin: resample_cmds[env_ids] at resampling_time
            new: update nav_commands every step
        """
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        # self._resample_commands(env_ids, on_the_way=True)

        # pos in world -> pos in robot
        pos_diff = self.position_targets - self.root_states[:, 0:3]
        self.goal_base = quat_rotate_inverse(self.base_quat, pos_diff)

        # update distance and reach_goal
        self.distance = torch.norm(self.root_states[:, :2] - self.position_targets[:, :2], dim=1)
        self.reach_goal = self.distance < (self.sigma)

        self.P_camera, self.P_image = self.camera_sensor.transform(self.goal_base)
        self.out_of_view = (self.P_image == -1).any(dim=-1)
        self.depth = torch.where(
            self.out_of_view,
            torch.ones_like(self.P_camera[:, -1]) * -1,  # set to -1 if out of view
            self.P_camera[:, -1]
        )

        # self.nav_commands = cart2polar(goal_base[:, :2])  # theta, rho
        env_ids = (self.episode_length_buf % int(self.cfg.commands.delay_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self.nav_commands[env_ids] = torch.cat([
            self.P_image[env_ids],
            self.depth[env_ids].unsqueeze(1)
        ], dim=-1)  # (img_x, img_y, distance)

        self.nav_commands_buffer[env_ids] = torch.where(
            (self.episode_length_buf[env_ids] <= 1)[:, None, None],
            torch.stack([self.nav_commands[env_ids]] * self.nav_commands_buffer[env_ids].shape[1], dim=1),
            torch.cat([
                self.nav_commands_buffer[env_ids, 1:],
                self.nav_commands[env_ids].unsqueeze(1)
            ], dim=1)
        )  

        self.nav_actions_buffer = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([self.nav_actions] * self.nav_actions_buffer.shape[1], dim=1),
            torch.cat([
                self.nav_actions_buffer[:, 1:],
                self.nav_actions.unsqueeze(1)
            ], dim=1)
        ) 

    def _resample_commands(self, env_ids, on_the_way=False):
        """ Only resample in reset (time out)
        """
        enbale_rand = True
        if len(env_ids) > 0:
            if on_the_way:
                # simulate a tracking jitter
                sim_move_pos = torch_rand_float(-1.0, 1.0, (len(env_ids), 2), device=self.device)
                self.position_targets[env_ids, :2] += sim_move_pos * enbale_rand
                sim_up_down_pos = torch_rand_float(-0.5, 0.5, (len(env_ids), 1), device=self.device)
                self.position_targets[env_ids, 2:3] += sim_up_down_pos * enbale_rand
                self.position_targets[env_ids, 2:3] = torch.clip(self.position_targets[env_ids, 2:3], 0.0, 0.2)
                
            else:
                _target_x = torch_rand_float(0.3, 5.0, (len(env_ids), 1), device=self.device)
                _target_y = torch_rand_float(-2.0, 2.0, (len(env_ids), 1), device=self.device)
                _target_z = torch_rand_float(-0.0, 0.2, (len(env_ids), 1), device=self.device) # simulate object on the ground
                self.position_targets[env_ids, 0:1] = self.env_origins[env_ids, 0:1] + _target_x * enbale_rand
                self.position_targets[env_ids, 1:2] = self.env_origins[env_ids, 1:2] + _target_y * enbale_rand
                self.position_targets[env_ids, 2:3] = self.env_origins[env_ids, 2:3] + _target_z * enbale_rand

    def compute_observations(self):
        """ Computes observations for updating nav agent
        """
        obs_buf = torch.cat([
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.nav_commands, 
            self.nav_actions
            ], dim=-1)

        self.obs_hist_buffer = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
            torch.cat([
                self.obs_hist_buffer[:, 1:],
                obs_buf.unsqueeze(1)
            ], dim=1)
        )
        
        self.obs_buf = self.obs_hist_buffer.view(self.num_envs, -1)

    def _draw_debug_vis(self):
        """ Draw position targets
        """
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_blue = gymutil.WireframeSphereGeometry(0.10, 8, 8, None, color=(0, 0, 1))
        for i in range(self.num_envs):
            x = self.position_targets[i, 0]
            y = self.position_targets[i, 1]
            z = self.position_targets[i, 2]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_blue, self.gym, self.viewer, self.envs[i], sphere_pose) 

        self.camera_sensor.visualize_img_coords(P_base=self.goal_base[0], P_camera=self.P_camera[0], P_image=self.P_image[0])
        # if hasattr(self, 'image'):
            # image = self.image.detach().cpu().numpy()
            # self.camera_sensor.visualize_img(image)
            # self._draw_camera_position()

    def _draw_camera_position(self):
        sphere_red = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(1, 0, 1))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3])
            camera_points = quat_apply(self.base_quat[i], self.camera_position)
            x = camera_points[0] + base_pos[0]
            y = camera_points[1] + base_pos[1]
            z = camera_points[2] + base_pos[2]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_red, self.gym, self.viewer, self.envs[i], sphere_pose) 

    # ------------- Cameras -------------
    def attach_camera(self, env_handle, actor_handle):
        if not hasattr(self, 'camera_position') or not hasattr(self, 'camera_angle'):
            self.camera_position = torch.tensor(self.cfg.camera_sensor.extrinsics.translation, device=self.device, dtype=torch.float)
            self.camera_angle = self.cfg.camera_sensor.extrinsics.angles
            self.enable_camera = self.cfg.camera_sensor.enable_camera
        if not self.enable_camera:
            return
        camera_props = gymapi.CameraProperties()
        camera_props.enable_tensors = True
        camera_props.width = self.cfg.camera_sensor.img_width
        camera_props.height = self.cfg.camera_sensor.img_height
        camera_props.horizontal_fov = self.cfg.camera_sensor.intrinsics.horizontal_fov
        camera_handle = self.gym.create_camera_sensor(
            env_handle, camera_props)
        root_handle = self.gym.get_actor_root_rigid_body_handle(
            env_handle, actor_handle)
        local_transform = gymapi.Transform()
        local_transform.p = gymapi.Vec3(*self.camera_position)
        local_transform.r = gymapi.Quat.from_euler_zyx(
            np.radians(self.camera_angle[0]), np.radians(self.camera_angle[1]), np.radians(self.camera_angle[2]))

        self.gym.attach_camera_to_body(
            camera_handle, env_handle, root_handle, local_transform, gymapi.FOLLOW_TRANSFORM)

        self.cam_handles.append(camera_handle)
    
    def update_first_image(self):
        if not self.enable_camera:
            return
        self.gym.step_graphics(self.sim)  # required to render in headless mode
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        image_ = self.gym.get_camera_image_gpu_tensor(self.sim,
                                                        self.envs[0],
                                                        self.cam_handles[0],
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
        # print(f"projected_gravity: {self.projected_gravity}")
        return torch.square(self.projected_gravity[:, 1])
    
    def _reward_nav_action_rate(self):
        return torch.square(torch.norm(self.nav_actions - self.last_nav_actions, dim=-1))

    def _reward_nav_action_limit(self):
        return torch.square(torch.norm(self.nav_actions - self.nav_actions_before_clip, dim=-1))

    def _reward_heading_target(self):
        forward = quat_apply(self.base_quat, self.forward_vec)
        xy_dif = self.position_targets[:,:2] - self.root_states[:, :2]
        xy_dif = xy_dif / (0.001 + torch.norm(xy_dif, dim=1).unsqueeze(1))
        # theta_error: 
        dir_cos = forward[:,0] * xy_dif[:,0] + forward[:,1] * xy_dif[:,1]
        # theta_error = torch.acos(dir_cos)
        theta_error = 1.0 - dir_cos
        return torch.exp(-theta_error / 0.01) * (dir_cos > 0.99) + 5.0 * self.reach_goal
        # return torch.exp(-theta_error / 0.01) * (~self.reach_goal)  + 2.0 * self.reach_goal
    
    def _reward_view_missing(self):
        """ Reward for missing the goal position in the view
        """
        return self.out_of_view.float()

    def _reward_lin_vel_y(self):
        return torch.abs(self.base_lin_vel[:, 1])
    
    # def _reward_tracking_view_center(self):
    #     """ Reward for align the goal position with the center of the view
    #     """
    #     align_error = torch.square((self.P_image[:, 0] - 0.5)) + torch.square((self.P_image[:, 1] - 0.5))
    #     return torch.exp(-align_error / 0.01)
    
    def _reward_tracking_horizontal_distance(self):
        """ Reward for tracking the horizontal distance to the goal position
        """
        dy_error = torch.square(self.goal_base[:, 1])
        return torch.exp(-dy_error / 0.01)

    def _reward_horizontal_distance_error(self):
        # Penalize large horizontal distance error
        target_grasp_width = 0.1
        dy_error = torch.abs(self.goal_base[:, 1]) - target_grasp_width
        dy_error = torch.clip(dy_error, min=0.0)
        return torch.square(dy_error)
    
    def _reward_keep_forward(self):
        # robot keep move at speed 
        target_vx = 0.35 # m/s
        lin_vel_error = torch.square(target_vx - self.base_lin_vel[:, 0])
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_reach_grasp_area(self):
        target_grasp_width = 0.1
        grasp_area = (torch.abs(self.goal_base[:, 1]) < (target_grasp_width/2.0))
        return self.reach_goal * grasp_area.float()
