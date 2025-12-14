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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgPPO
from legged_gym.envs.base.legged_robot_nav_config import LeggedRobotNavCfg
from legged_gym.envs.base.target_config import TargetCfg

# NUM_NAV_COMMANDS = 2  # P_img_x, P_img_y
NUM_SIGMA_POINTS = 5
NUM_NAV_COMMANDS = 15 # 5 points * 3 coords (x, y, z) in Base Frame
EPISODE_LENGTH_S = 12
USE_RNN = False

class Go2NavFlatCfg( LeggedRobotNavCfg ):
    target = TargetCfg
    debug_viz = True
    pixel_gain = 10.0
    class env(LeggedRobotNavCfg.env):
        num_position = 3 # x, y, z
        num_nav_actions = 4 # vx, vy, vyaw, pitch
        nav_history_len = 10
        history_len = 5
        # num_priv = 3 + 1 # +3 for P_base, +1 for timer
        # num_priv = 3
        num_sigma_points = NUM_SIGMA_POINTS
        num_nav_commands = NUM_NAV_COMMANDS # 5 points * 2 (u, v) + 1 (distance)
        # num_props = num_nav_actions + 11 # lin_vel(3), ang_vel(3), gravity(3), pitch(1), phase(1)
        # num_props = num_nav_actions + num_nav_commands + 10 # lin_vel(3), ang_vel(3), gravity(3), pitch(1)
        num_props = num_nav_actions + num_nav_commands + 12 # lin_vel(3), ang_vel(3), gravity(3), rpy(3)
        
        # num_priv = 3 + 1 # +3 for P_base, +1 for timer
        num_priv = NUM_SIGMA_POINTS * 3 # P_camera [N, M, 3] flattened
        
        if USE_RNN:
            num_observations = num_props * history_len
            num_privileged_obs = num_props * history_len + num_priv
        else:
            num_observations = num_nav_commands * nav_history_len + num_props * history_len + num_priv
            num_privileged_obs = None

        num_envs = 2048
        episode_length_s = EPISODE_LENGTH_S # episode length in seconds  # will be randomized in [s-minus, s]
        no_nav = True
        fear_ctrl_heading = False
        curriculum_episode_length_s = False
        debug_viz = False
        explore = False
        arbiter = False
        encode = True
        no_time_limit = False

    class init_state( LeggedRobotNavCfg.init_state ):
        pos = [0.0, 0.0, 0.42]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }
    

    class commands:
        add_boost = False
        img_frame = True  # if True, the commands are given in the image frame, otherwise in the camera frame
        curriculum = False
        max_curriculum = 1.
        num_commands = 4
        num_nav_commands = NUM_NAV_COMMANDS
        resampling_time = 2.0
        start_resampling_time = 3.0
        end_resampling_time = EPISODE_LENGTH_S - 2.0
        period_s = 0.2  # [s] time period for timing-based commands
        enable_delay = False
        max_delay_time_ms = 50  # maximum delay time in milliseconds
        min_delay_time_ms = 0  # minimum delay time in milliseconds

        class ranges:
            limit_vx = [0.18, 0.5]  # [m/s]
            limit_vy = [-0.1, 0.1]  # [m/s]
            limit_vyaw = [-1.0, 1.0]  # [rad/s]
            limit_pitch = [-3.14/6, 3.14/6]  # [rad]
            heading = [-0.3, 0.3]  # a residual heading plus theta
    
    class camera_sensor:
        enable_camera = True  # if True, the image of isaacgym is enabled, otherwise it is disabled
        save_debug_images = False # if True, save debug images to disk, otherwise view in real-time
        vis_target_points = False # if True, visualize target points (green)
        vis_sigma_3d = False # if True, visualize 3D sigma points (blue)
        vis_sigma_2d = True # if True, visualize 2D sigma points (yellow)
        fix_extrinsics = True  # if True, the camera extrinsics are fixed, otherwise they are randomized
        fix_intrinsics = True  # if True, the camera intrinsics are fixed, otherwise they are randomized
        fix_img_shape = True  # if True, the image shape is fixed, otherwise it is randomized
        clip_invalid = False # if True, the invalid image coordinates are clipped to -1, otherwise they are kept as is
        max_out_of_view_duration = 2.0 # [s] the duration to keep the out of view coordinates
        enable_out_of_view_drift = True # if True, add random walk drift when out of view
        drift_scale = 0.02 # scale of the random walk drift per step

        class intrinsics: # Intrinsics parameters
            # Zed mini, HD720 mode
            # img_width = 1280
            # img_height = 720
            # horizontal_fov = 82.33
            # fx = 731.995849609375
            # fy = 731.995849609375
            # cx = 620.0855102539062
            # cy = 362.5731201171875

            # # Zed mini, VGA mode
            # img_width = 672
            # img_height = 376
            # horizontal_fov = 85.0
            # fx = 367.0 # fx = img_width / (2 * tan(horizontal_fov/2 * pi/180))
            # fy = 367.0 # fy = fx
            # cx = 336.0 # cx = img_width / 2
            # cy = 188.0 # cy = img_height / 2

            # # Zed mini, HD720 mode, scaled to 320x180
            # img_width = 320
            # img_height = 180
            # horizontal_fov = 82.33
            # fx = 182.9919 # fx = img_width / (2 * tan(horizontal_fov/2 * pi/180))
            # fy = 182.9919 # fy = fx
            # cx = 155.02 # 160.0
            # cy = 90.64 # 90.0

            # Realsense D435i
            # 640x360, HFOV=70.26
            # img_width = 640
            # img_height = 360
            horizontal_fov = 70.26
            # fx = 454.768310546875  # fx = img_width / (2 * np.tan(np.deg2rad(horizontal_fov) / 2))
            # fy = 454.4901123046875
            # cx = 325.7699279785156
            # cy = 184.68618774414062


            # 320*180
            # img_width = 320
            # img_height = 180
            # fx = 227.3841552734375
            # fy = 227.24505615234375
            # cx = 162.8849639892578
            # cy = 92.34309387207031

            # 320*240
            img_width = 320
            img_height = 240
            fx = 303.1788635253906
            fy = 302.993408203125
            cx = 163.8466033935547
            cy = 123.1241226196289

            horizontal_fov_range = [-2.0, 2.0] # [degree]
            img_height_range = [90, 720] # [pixel]
            img_width_range = [160, 1280] # [pixel]

        class extrinsics: # Extrinsics parameters
            #  ================= fixed extrinsics =================
            translation = [0.305, 0.017, 0.128]  # Translation: forward, left, upward
            angles = [0.0, 30.0, 0.0]  # Euler angles: yaw, pitch, roll
            
            #  ================= random extrinsics =================
            # Randomization ranges around the fixed extrinsics
            yaw_range = [-0.5, 0.5]   # [degree]
            pitch_range = [-2.0, 10.0] # [degree]
            roll_range = [-0.5, 0.5]  # [degree]

            dx_range = [-0.01, 0.01]   # [m]
            dy_range = [-0.01, 0.01]   # [m]
            dz_range = [-0.01, 0.01]   # [m]

    class control( LeggedRobotNavCfg.control ):
        # PD Drive parameters:
        control_type = 'P'

        stiffness = {'joint': 30.}  # [N*m/rad]
        damping = {'joint': 0.75}     # [N*m*s/rad
            
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotNavCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2_description/urdf/go2_description.urdf'
        flip_visual_attachments = True
        fix_base_link = False
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "Head_upper", "Head_lower", "base"] # collision reward
        terminate_after_contacts_on = ["base", "Head_upper", "Head_lower"] # termination
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
    

    class terrain( LeggedRobotNavCfg.terrain ):
        mesh_type = 'plane'
        terrain_types = ['flat','rough']  # do not duplicate!
        terrain_proportions = [0.8, 0.2]
        num_rows = 10 # number of terrain rows (levels)
        num_cols = 10 # number of terrain cols (types)
        measure_heights = True

    class domain_rand( LeggedRobotNavCfg.domain_rand ):
        randomize_friction = True
        friction_range = [-0.2, 2.5]
        randomize_restitution = True
        restitution_range = [0.0, 1.0]
        randomize_base_mass = True
        added_mass_range = [-1., 2.0]
        randomize_base_com = True
        added_com_range = [-0.05, 0.05]
        push_robots = True
        push_interval_s = 5
        c = 0.5
        roll_robots = False
        max_vel_roll = 1.57 / 3  # [rad/s]
        roll_interval = 0.4 / 0.02

        randomize_yaw = False
        randomize_pitch = False
        randomize_roll = False
        init_yaw_range = [-3.14, 3.14]
        init_pitch_range = [-0.1, 0.1]
        init_roll_range = [-0.1, 0.1]

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 2.0
            pitch = 1.0
            euler_rpy = 1.0

        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        add_camera_noise = True
        noise_level = 1.0
        invalid_depth_interval_s = 2.0
        invalid_p_img_interval_s = 5.0
        class noise_scales:
            dof_pos = 0.01 # 0.01 
            dof_vel = 1.0 # 1.0
            lin_vel = 0.1 # 0.1
            ang_vel = 0.1 # 0.2
            gravity = 0.05 # 0.05
            pitch = 0.1 # 0.1
            euler_rpy = 0.1
            
            P_img_u = 0.01
            P_img_v = 0.01
            P_img_depth = 0.2

    class rewards():
        class scales():
            heading_target = 2.0 # 1.0
            lin_vel_z = -1.0 # -3.0
            ang_vel_xy = -0.1
            orientation_y = -4.0
            nav_action_rate = -2.0
            nav_action_limit = -2.0
            view_missing = -2.0
            tracking_horizontal_distance = 50.0
            tracking_view_center = 0.5
            horizontal_distance_error = -0.0
            forward = 1.0 # 1.0
            reach_grasp_area = 0

            stand_still = 0 # 500.0
            action_rate = 0.0 # -0.01 # -0.005 
            cmds_track = 0.0 # -0.2 
            torques = 0.0 #  -0.0002
            reach_target = 0.0 # 50.0 

        soft_dof_pos_limit = 0.95
        base_height_target = 0.25
        only_positive_rewards = False
        position_target_sigma_soft = 2.0
        position_target_sigma_tight = 0.5
        heading_target_sigma = 1.0
        rew_duration = 2.0 # 2.0
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.85
        max_contact_force = 100.
        tracking_sigma = 0.1

class Go2NavFlatCfgPPO( LeggedRobotCfgPPO ):
    runner_class_name = 'OnPolicyRunner'
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        # entropy_coef = 0.05
        # entropy_coef = 0.003
        entropy_coef = 0.01

        
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'go2_nav_flat'

        save_interval = 200  # save model every n iterations
        max_iterations = 4000  # maximum number of training iterations
        
        # policy_class_name = 'ActorCriticRnn'
        if USE_RNN:
            policy_class_name = 'ActorCriticRecurrent'
        else:
            policy_class_name = 'ActorCriticEncoder'
            # policy_class_name = "ActorCriticRecurrentEncoder"
        algorithm_class_name = 'PPO'

    class policy( LeggedRobotCfgPPO.policy ):
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        rnn_type = 'gru'

        # actor_hidden_dims = [256, 128, 64]
        # critic_hidden_dims = [256, 128, 64]
        # rnn_hidden_size = 128

        # actor_hidden_dims = [128, 64, 32]
        # critic_hidden_dims = [128, 64, 32]
