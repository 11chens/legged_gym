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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
import sys
import numpy as np
import argparse
import torch

parser = argparse.ArgumentParser(description="Run the Go2 robot in navigation environment.")
parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--load_run", type=str,  help="Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."),
parser.add_argument("--checkpoint", type=int,  help="Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided.")
args = parser.parse_args()

if args.debug:
    import debugpy

    ip_address = ("0.0.0.0", 6666)
    print(f"Process: {sys.argv[:]}")
    print(f"Is waiting for attach at {ip_address[0]}:{ip_address[1]}", flush=True)
    debugpy.listen(ip_address)
    debugpy.wait_for_client()
    debugpy.breakpoint()


def play(args):
    env: LeggedRobotNav
    env_cfg: Go2NavFlatCfg

    # args.load_run = '12_08_22-24-56_'
    # args.checkpoint = 1000
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = 1
    env_cfg.debug_viz = True
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.camera_sensor.fix_extrinsics = True
    env_cfg.camera_sensor.fix_intrinsics = True
    env_cfg.camera_sensor.fix_img_shape = True
    env_cfg.camera_sensor.enable_camera = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # priv_obs = env.get_privileged_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env,
                                                          name=args.task,
                                                          args=args,
                                                          train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    if EXPORT_ONNX:
        # onnx_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs',
        #                     train_cfg.runner.experiment_name, 'exported')
        onnx_dir = '/home/robot/Data/onboard_data/onnx_models/homi/nav_model'
        ppo_runner.alg.actor_critic.export_onnx_model(onnx_dir=onnx_dir)

    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(
        env_cfg.viewer.pos)
    env.set_camera(camera_position, camera_position + camera_direction)

    # TODO: video recording
    for i in range(20 * int(env.max_episode_length)):

        actions = policy(obs.detach())
        obs, priv_obs, rews, dones, infos = env.step(actions.detach())

        dx = env.P_base[0, 0].item()
        dy = env.P_base[0, 1].item()
        dz = env.P_base[0, 2].item()

        cx = env.nav_actions[0, 0].item()
        cy = env.nav_actions[0, 1].item()
        cyaw = env.nav_actions[0, 2].item()
        cpitch = env.nav_actions[0, 3].item()

        u_c = env.nav_commands[0, 0].item()
        v_c = env.nav_commands[0, 1].item()
        depth_c = env.nav_commands[0, 2].item()

        u = env.P_image[0, 0].item()
        v = env.P_image[0, 1].item()
        depth = env.depth[0].item()

        vx = env.base_lin_vel[0, 0]
        vy = env.base_lin_vel[0, 1]
        vyaw = env.base_ang_vel[0, 2]
        pitch = env.euler_rpy[0, 1]

        distance = env.distance[0].item()

        # print(f"vel: ({vx:.2f}, {vy:.2f}, {vyaw:.2f}, {pitch:.2f})")
        # print(f"P_img: (u:{u:.2f}, v:{v:.2f}, d:{depth:.2f})")
        print(f"Command: (u_c:{u_c:.2f}, v_c:{v_c:.2f}, d_c:{depth_c:.2f})")
        print(f"Action: (cx: {cx:.2f}, cy: {cy:.2f}, cyaw: {cyaw:.2f}, cpitch: {cpitch:.2f})")
        print(f"Base: ({vx:.2f}, {vy:.2f}, {vyaw:.2f}, {pitch:.2f})")
        print(f"Distance: ({distance:.2f})")
        
if __name__ == '__main__':
    EXPORT_ONNX = True
    args = get_args(args)
    play(args)
