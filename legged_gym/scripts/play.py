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
import onnxruntime as ort
import time
import cv2
from isaacgym import gymapi

parser = argparse.ArgumentParser(description="Run the Go2 robot in navigation environment.")
parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--load_run", type=str,  help="Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."),
parser.add_argument("--checkpoint", type=int,  help="Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided.")
parser.add_argument("--onnx", action="store_true", help="Export and run the ONNX model.")
parser.add_argument("--video", action="store_true", help="Record video during play.")
parser.add_argument("--video_fpv", action="store_true", help="Record FPV video during play.")

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

    args.load_run = '04_13_18-39-16_'
    # args.checkpoint = 10000
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = 1

    env_cfg.debug_viz = False
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.terrain_proportions = [1.0]
    env_cfg.noise.add_noise = True

    env_cfg.camera_sensor.fix_extrinsics = True
    env_cfg.camera_sensor.fix_intrinsics = True
    env_cfg.camera_sensor.fix_img_shape = True
    env_cfg.camera_sensor.enable_camera = True
    env_cfg.camera_sensor.num_vis_points = 200
    env_cfg.commands.resample.replay_failed_prob = 0.7
    env_cfg.commands.enable_delay = False

    env_cfg.commands.frame_drop_prob = 0.0 # Disable random frame drops causing spikes
    env_cfg.commands.nav_refresh_steps_range = [1, 1] # Use fixed 50Hz for cleaner evaluation
    env_cfg.commands.hold_time_s = 1.0 # 1 second hold time to consider as success

    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.friction_range = [5.0, 10.0]

    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False

    env_cfg.commands.enable_out_of_view_drift = True # if True, add random walk drift when out of view
    env_cfg.commands.resample.adjust_obj_pose = True
    env_cfg.commands.resample.enable_success_feeding = True # If True, use success  feeding mechanism
    env_cfg.commands.resample.feeding_prob = 0.33
    env_cfg.commands.resample.force_look_upwards = False
    env_cfg.commands.resample.ranges.min_dist = 2.0
    env_cfg.commands.resample.ranges.max_dist = 3.0

    env_cfg.target.init.place_prob = 0.0 # 0 is pick only, 1 is place only, in between is a mix
    env_cfg.target.init.vertical_prob = 0.0
    env_cfg.target.perception.add_pre_pca_noise = True
    env_cfg.target.perception.alpha_range = [1.0, 1.0] # Sigma points scaling factor range
    env_cfg.target.shape.types = ["sphere"]
    # env_cfg.target.shape.types = ["ycb"]
    # env_cfg.target.shape.types = ["box"] # place prob is 1
    # env_cfg.target.shape.types = ["cuboid"]

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

    if args.onnx:
        onnx_dir = os.path.expanduser("~/Data/onboard_data/onnx_models/homi/nav_model")
        os.makedirs(onnx_dir, exist_ok=True)
        ppo_runner.alg.actor_critic.export_onnx_model(onnx_dir=onnx_dir)
        
        # Load ONNX Model
        onnx_path = os.path.join(onnx_dir, "model.onnx")
        print(f"Loading ONNX model from {onnx_path}")
        ort_sess = ort.InferenceSession(onnx_path)
        
        # Get model info for hidden state init
        model = ppo_runner.alg.actor_critic

        if model.is_recurrent:
            rnn_layers = model.memory_a.rnn.num_layers
            rnn_hidden = model.memory_a.rnn.hidden_size
            # Init Hidden States (Batch size 1)
            h_state = np.zeros((rnn_layers, 1, rnn_hidden), dtype=np.float32)

    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(
        env_cfg.viewer.pos)
    env.set_camera(camera_position, camera_position + camera_direction)

    # Setup Camera for Video Recording (Fix Top-Down View)
    camera_props = gymapi.CameraProperties()
    camera_props.width = 2048
    camera_props.height = 2048
    
    # Initial Camera Position (will be updated dynamically)
    cam_offset = np.array([1.0, 1.0, 3.0]) # 3m above robot
    cam_handle = env.gym.create_camera_sensor(env.envs[0], camera_props)
    # Update Camera Position to Follow Robot (Set before next step's render or use for current capture)
    robot_pos = env.root_states[0, :3].cpu().numpy()
    cam_pos = gymapi.Vec3(robot_pos[0] + cam_offset[0], robot_pos[1] + cam_offset[1], robot_pos[2] + cam_offset[2])
    cam_target = gymapi.Vec3(robot_pos[0], robot_pos[1] + 0.001, robot_pos[2])
    env.gym.set_camera_location(cam_handle, env.envs[0], cam_pos, cam_target)
    
    log_data = []
    video_writer = None
    video_fpv_writer = None
    episode = 0
    
    # Initialize main video writer if pre-defined dimensions are used
    if args.video:
        os.makedirs("logs", exist_ok=True)
        video_filename = os.path.expanduser(f"logs/top_{env_cfg.target.shape.types[0]}.mp4")
        print(f"Recording video to {video_filename}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Match preference
        video_writer = cv2.VideoWriter(video_filename, fourcc, 50.0, (camera_props.width, camera_props.height))

    for i in range(20 * int(env.max_episode_length)):
        object_pos = env.object_pos[0]

        if args.onnx:
            priv, obs = ppo_runner.alg.actor_critic.get_priv_separated(obs)
            if model.is_recurrent:
                if hasattr(ppo_runner.alg.actor_critic, 'get_current_frame'):
                    current_prop = ppo_runner.alg.actor_critic.get_current_frame(obs)
                else:
                    current_prop, _ = ppo_runner.alg.actor_critic.extract_obs(obs)
                
                obs_cpu = current_prop.detach().cpu().numpy()
                ort_inputs = {'obs': obs_cpu, 'h_in': h_state}
                ort_outs = ort_sess.run(None, ort_inputs)
                action_np = ort_outs[0]
                h_state = ort_outs[1]
            else:
                obs_cpu = obs.detach().cpu().numpy()
                ort_inputs = {'obs': obs_cpu}
                ort_outs = ort_sess.run(None, ort_inputs)
                action_np = ort_outs[0]

            actions = torch.from_numpy(action_np).to(env.device)
        else:
            actions = policy(obs.detach())
            
        obs, priv_obs, rews, dones, infos = env.step(actions.detach())
        
        # Sync and update graphics
        env.gym.fetch_results(env.sim, True)
        env.gym.step_graphics(env.sim)
        
        # Position tracing for overarching camera view
        if args.video:
            robot_pos = env.root_states[0, :3].cpu().numpy()
            obj_pos = object_pos.cpu().numpy()
            cam_pos = gymapi.Vec3(obj_pos[0] + cam_offset[0], obj_pos[1] + cam_offset[1], obj_pos[2] + cam_offset[2])
            cam_target = gymapi.Vec3(obj_pos[0], obj_pos[1] + 0.001, obj_pos[2])
            env.gym.set_camera_location(cam_handle, env.envs[0], cam_pos, cam_target)
            
        # Render sensors after camera location update
        env.gym.render_all_camera_sensors(env.sim)
        
        # Get frame from standard camera if args.video
        if args.video:
            img_rgba = env.gym.get_camera_image(env.sim, env.envs[0], cam_handle, gymapi.IMAGE_COLOR)
            img = img_rgba.reshape((camera_props.height, camera_props.width, 4))[:, :, :3]
            # Convert RGB to BGR for OpenCV VideoWriter
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video_writer.write(img_bgr)

        # FPV recording logic (also updates viewer debug lines using default visual downsampling)
        fpv_frame = env._draw_debug_vis(force_fpv=args.video_fpv)
        
        if args.video_fpv and fpv_frame is not None:
            if video_fpv_writer is None:
                video_fpv_filename = os.path.expanduser(f"logs/fpv_{env_cfg.target.shape.types[0]}.mp4")
                print(f"Recording FPV video to {video_fpv_filename}")
                h, w, _ = fpv_frame.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_fpv_writer = cv2.VideoWriter(video_fpv_filename, fourcc, 50.0, (w, h))
            video_fpv_writer.write(fpv_frame)

        episode += dones.sum().item()
        
        # Stop condition: Record for specific episodes
        if episode == 10:
            print("Completed 10 episodes. Saving data...")
            if video_writer is not None:
                video_writer.release()
                print("Video saved.")
            if video_fpv_writer is not None:
                video_fpv_writer.release()
                print("FPV Video saved.")
            break

if __name__ == '__main__':
    args = get_args(args)
    play(args)

