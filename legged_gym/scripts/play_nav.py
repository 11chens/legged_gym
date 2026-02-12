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
parser.add_argument("--npz", action="store_true", help="Record npz data during play.")

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

    # args.load_run = '02_11_21-08-28_'
    # args.checkpoint = 1000
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = 1
    env_cfg.env.sigma_frame = "camera" # "base", "camera", "image"

    env_cfg.debug_viz = False
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
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
    env_cfg.domain_rand.friction_range = [0.2, 0.6]

    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False

    env_cfg.commands.enable_out_of_view_drift = True # if True, add random walk drift when out of view
    env_cfg.commands.resample.adjust_obj_pose = True
    env_cfg.commands.resample.enable_success_feeding = True # If True, use success  feeding mechanism
    env_cfg.commands.resample.feeding_prob = 0.33 # If True, use success  feeding mechanism
    env_cfg.commands.resample.force_look_upwards = False
    env_cfg.commands.resample.ranges.min_dist = 2.0
    env_cfg.commands.resample.ranges.max_dist = 3.0
    # env_cfg.commands.drift_scale = 0.0
            

    # env_cfg.noise.noise_scales.nav_pos_3d = [0.0, 0.0, 0.0] # Center noise [m] in Camera Frame (X, Y, Z)
    # env_cfg.noise.noise_scales.nav_scale_3d = 0.0 # Scale noise (proportional)
    # env_cfg.noise.noise_scales.nav_rot_3d = 0.1 # Rotation noise [rad] (~15 deg)

    env_cfg.target.init.place_prob = 0.0
    env_cfg.target.init.vertical_prob = 0.0
    env_cfg.target.perception.add_pre_pca_noise = True
    env_cfg.target.perception.alpha_range = [1.0, 1.0] # Sigma points scaling factor range
    env_cfg.target.shape.types = ["ycb"]
    # env_cfg.target.shape.types = ["sphere"]
    # env_cfg.target.shape.types = ["box"]
    # env_cfg.target.shape.types = ["cuboid"]
    # env_cfg.target.shape.dims_range = [[0.05, 0.10], [0.05, 0.10], [0.05, 0.10]] # longer pick cuboid
    # env_cfg.target.shape.dims_range = [[0.05, 0.08], [0.05, 0.08], [0.05, 0.08]] # little pick cuboid
    # env_cfg.target.shape.dims_range = [[0.04, 0.06], [0.04, 0.06], [0.04, 0.06]] # little box
    # env_cfg.target.shape.box_dims_range = [[0.03, 0.2], [0.03, 0.2], [0.2, 0.35]] # long box
    # env_cfg.target.shape.box_dims_range = [[0.03, 0.05], [0.03, 0.05], [0.03, 0.05]] # little place box
    # env_cfg.target.shape.box_dims_range = [[0.03, 0.05], [0.03, 0.05], [0.2, 0.35]] # thin bucket
    # env_cfg.target.shape.box_dims_range = [[0.25, 0.25], [0.35, 0.35], [0.15, 0.16]] # big box

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
        # onnx_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs',
        #                     train_cfg.runner.experiment_name, 'exported')
        onnx_dir = '/home/robot/Data/onboard_data/onnx_models/homi/nav_model'
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
    cam_offset = np.array([0.0, 0.0, 3.0]) # 3m above robot
    cam_handle = env.gym.create_camera_sensor(env.envs[0], camera_props)
    # Update Camera Position to Follow Robot (Set before next step's render or use for current capture)
    robot_pos = env.root_states[0, :3].cpu().numpy()
    cam_pos = gymapi.Vec3(robot_pos[0] + cam_offset[0], robot_pos[1] + cam_offset[1], robot_pos[2] + cam_offset[2])
    cam_target = gymapi.Vec3(robot_pos[0], robot_pos[1] + 0.001, robot_pos[2])
    env.gym.set_camera_location(cam_handle, env.envs[0], cam_pos, cam_target)
    
    log_data = []
    start_record = False
    video = None
    video_fpv = None
    episode = 0
    
    # TODO: video recording
    for i in range(20 * int(env.max_episode_length)):
        # time.sleep(0.05) # slow down for visualization
        env.alpha *= 0.0 
        env.alpha += 0.2
        object_pos = env.object_pos[0]

        if args.onnx:
            # Prepare inputs
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
        
        # Sync and Render for the custom sensor explicitly to ensure it's updated
        env.gym.fetch_results(env.sim, True)
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)

        # Draw and capture FPV frame (also updates viewer debug lines)
        # If recording FPV, we force every frame. Otherwise defaults to visual downsampling.
        fpv_frame = env._draw_debug_vis(force_fpv=args.video_fpv)
        
        if episode == 8 and not start_record:
            if args.npz:
                start_record = True
                i_now = i
                print("Starting log recording...")
            # Initialize Video Writer
            if args.video or args.video_fpv:
                start_record = True
                i_now = i
                
                cam_target = gymapi.Vec3(object_pos[0], object_pos[1] + 0.001, object_pos[2])
                env.gym.set_camera_location(cam_handle, env.envs[0], cam_pos, cam_target)

                if args.video:
                    video_filename = os.path.expanduser("~/nav_recording.mp4")
                    print(f"Recording video to {video_filename}")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Match play_room's uppercase lowercase preference
                    # Assuming 50fps for playback speed
                    video = cv2.VideoWriter(video_filename, fourcc, 50.0, (camera_props.width, camera_props.height))

                if args.video_fpv:
                    # FPV video will be initialized on first frame to get dimensions
                    video_fpv_filename = os.path.expanduser("~/nav_fpv_recording.mp4")
                    print(f"Recording FPV video to {video_fpv_filename}")

        if start_record:
            if args.video:
                # Capture and write frame
                img_rgba = env.gym.get_camera_image(env.sim, env.envs[0], cam_handle, gymapi.IMAGE_COLOR)
                img = img_rgba.reshape((camera_props.height, camera_props.width, 4))[:, :, :3]
                # Convert RGB to BGR for OpenCV VideoWriter
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                video.write(img_bgr)
            
            if args.video_fpv and fpv_frame is not None:
                if video_fpv is None:
                    h, w, _ = fpv_frame.shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_fpv = cv2.VideoWriter(video_fpv_filename, fourcc, 50.0, (w, h))
                video_fpv.write(fpv_frame)
            
            if i % 5 == 0:
                # Use dictionary for structured logging
                log_step = {
                    "base_lin_vel": env.base_lin_vel_pred[0].detach().cpu().numpy().flatten(), # 3
                    "base_ang_vel": env.base_ang_vel[0].detach().cpu().numpy().flatten(), # 3
                    "euler_rpy": env.euler_rpy[0].detach().cpu().numpy().flatten(), # 3
                    "projected_gravity": env.projected_gravity[0].detach().cpu().numpy().flatten(), # 3
                    "nav_commands": env.nav_commands[0].detach().cpu().numpy().flatten(), # 21
                    "task_flag": env.task_flags[0].detach().cpu().numpy().flatten(), # 1
                    "actions": actions[0].detach().cpu().numpy().flatten() # 4
                }

                log_data.append(log_step)
            
            if (i - i_now == 2):
                print(f"Start Pos: (base_x: {base_x:.2f}, base_y: {base_y:.2f}, base_z: {base_z:.2f})")

            
            # Stop condition: Record for 300 steps (approx 6s)
            # if (i - i_now == 300):
            if episode == 10:
                # Save log data
                if args.npz:
                    log_path = os.path.expanduser("~/sim_nav_log.npz")
                    save_dict = {k: np.array([step[k] for step in log_data]) for k in log_data[0].keys()}
                    np.savez(log_path, **save_dict)
                    print(f"Saved nav log data to {log_path} with {len(log_data)} points")
                if args.video:
                    # Release video
                    video.release()
                    print("Video saved.")
                if args.video_fpv:
                    if video_fpv is not None:
                        video_fpv.release()
                        print("FPV Video saved.")
                break # Exit after saving

        episode += dones.sum().item()

        dx = env.P_base[0, 0].item()
        dy = env.P_base[0, 1].item()
        dz = env.P_base[0, 2].item()

        cx = env.nav_actions[0, 0].item()
        cy = env.nav_actions[0, 1].item()
        cyaw = env.nav_actions[0, 2].item()
        cpitch = env.nav_actions[0, 3].item()

        ori_cx = env.orig_nav_actions[0, 0].item()
        ori_cy = env.orig_nav_actions[0, 1].item()
        ori_cyaw = env.orig_nav_actions[0, 2].item()
        ori_cpitch = env.orig_nav_actions[0, 3].item()



        main_x = env.sigma_points_camera[0, 0, 0].item()
        main_y = env.sigma_points_camera[0, 0, 1].item()
        main_z = env.sigma_points_camera[0, 0, 2].item()

        vx = env.base_lin_vel[0, 0]
        vy = env.base_lin_vel[0, 1]
        vyaw = env.base_ang_vel[0, 2]
        pitch = env.euler_rpy[0, 1]

        distance = env.distance[0].item()

        print(f"main sigma points: ({main_x}, {main_y}, {main_z})")

        # print(f"vel: ({vx:.2f}, {vy:.2f}, {vyaw:.2f}, {pitch:.2f})")
        # print(f"Command: (sig_x: {sig_x:.2f}, sig_y: {sig_y:.2f}, sig_z: {sig_z:.2f})")
        # print(f"Action: (cx: {cx:.2f}, cy: {cy:.2f}, cyaw: {cyaw:.2f}, cpitch: {cpitch:.2f})")
        # print(f"Orig Action: (cx: {ori_cx:.2f}, cy: {ori_cy:.2f}, cyaw: {ori_cyaw:.2f}, cpitch: {ori_cpitch:.2f})")
        # print(f"pitch: {pitch:.2f}")
        # print(f"Base: ({vx:.2f}, {vy:.2f}, {vyaw:.2f}, {pitch:.2f})")
        # print(f"Distance: ({distance:.2f})")


    # Optional: Plotting
    # script_path = "/home/robot/project/quad_deploy/scripts/plot_homi_nav_log.py"
    # os.system(f"python {script_path} {log_path}")
        
if __name__ == '__main__':
    args = get_args(args)
    play(args)
