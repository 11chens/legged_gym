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

NUM_NAV_COMMANDS = 3  # P_img_x, P_img_y, distance
# NUM_NAV_COMMANDS = 2  # P_img_x, P_img_y

class Go2NavFlatCfg( LeggedRobotNavCfg ):
    debug_viz = False
    class env(LeggedRobotNavCfg.env):
        num_position = 3 # x, y, z
        num_props = 9 # lin_vel, ang_vel, gravity
        num_nav_actions = 4 # vx, vy, vyaw, pitch
        history_len = 5
        num_observations = (NUM_NAV_COMMANDS + num_props + num_nav_actions) * history_len
        num_envs = 2048
        episode_length_s = 16 # episode length in seconds  # will be randomized in [s-minus, s]
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
        curriculum = False
        max_curriculum = 1.
        num_commands = 4
        num_nav_commands = NUM_NAV_COMMANDS
        resampling_time = 6
        delay_time = 0.1 # delay time in seconds
        class ranges:
            limit_vx = [-0.0, 1.0]  # [m/s]
            limit_vy = [-0.3, 0.3]  # [m/s]
            limit_vyaw = [-1.0, 1.0]  # [rad/s]
            limit_pitch = [-0.5, 0.5]  # [rad]

            # limit_vx = [0.35, 0.36]  # [m/s]
            limit_vy = [-0.05, 0.05]  # [m/s]
            # limit_vyaw = [-0.0, 0.0]  # [rad/s]

            use_polar = False
            # if use polar: it is rho and theta, else x and y
            pos_1 = [1.5, 7.5] # min max [m] 
            pos_2 = [-2.0, 2.0]  # rad if polar
            heading = [-0.3, 0.3]  # a residual heading plus theta
    
    class camera_sensor:
        enable_camera = False  # if True, the camera sensor is enabled, otherwise it is disabled
        fix_extrinsics = False  # if True, the camera extrinsics are fixed, otherwise they are randomized
        fix_intrinsics = True  # if True, the camera intrinsics are fixed, otherwise they are randomized
        img_width = 1280
        img_height = 720
        class intrinsics: # Intrinsics parameters
            # Zed mini, HD720 mode
            horizontal_fov = 82.33
            fx = 731.995849609375
            fy = 731.995849609375
            cx = 620.0855102539062
            cy = 362.5731201171875

        class extrinsics: # Extrinsics parameters
            translation = [0.5, 0.0, 0.0] # forward, left, upper
            angles = [0.0, 10.0, 0.0] # yaw, pitch, roll

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
        terminate_after_contacts_on = ["base", "Head_upper", "Head_lower"] # termination rewrad
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
    

    class terrain( LeggedRobotNavCfg.terrain ):
        mesh_type = 'plane'
        terrain_types = ['flat','rough']  # do not duplicate!
        terrain_proportions = [0.4, 0.6]
        num_rows = 10 # number of terrain rows (levels)
        num_cols = 10 # number of terrain cols (types)
        measure_heights = True

    class domain_rand:
        randomize_friction = True
        friction_range = [-0.2, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1.5, 1.5]
        randomize_dof_bias = True
        max_dof_bias = 0.08
        randomize_timer_minus = 2.0  # timer_left is initialized with randomization: U(T-this, T)

        push_robots = True
        push_interval_s = 2.5
        max_push_vel_xy = 0.0  # not used
        
        randomize_yaw = True
        randomize_yaw = True
        init_yaw_range = [-3.14, 3.14]
        randomize_roll = False
        randomize_pitch = False
        randomize_xy = True
        init_x_range = [-0.5, 0.5]
        init_y_range = [-0.5, 0.5]
        randomize_velo = False
        init_vlinx_range = [-0.5,0.5]
        init_vliny_range = [-0.5,0.5]
        init_vlinz_range = [-0.5,0.5]
        init_vang_range = [-0.5,0.5]
        randomize_init_dof = True
        init_dof_factor=[0.5, 1.5]
        stand_bias3 = [0.0, 0.0, 0.0]


    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 2.0
            pitch = 1.0
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0
        class noise_scales:
            dof_pos = 0.03 # 0.01 
            dof_vel = 1.75 #1.5
            lin_vel = 0.1 # 0.1
            ang_vel = 0.2 # 0.2
            gravity = 0.1 # 0.05
            height_measurements = 0.1
            ray2d = 0.2  # 2^0.2 = 1.1487
    

    class rewards():
        class scales():
            heading_target = 1.0 # 1.0
            lin_vel_z = -1.0 # -3.0 
            ang_vel_xy =  -0.2 
            orientation_y = -1.0
            nav_action_rate = 0.0 # -0.5
            nav_action_limit = -1.0
            view_missing = -0.5
            tracking_horizontal_distance = 1.0
            horizontal_distance_error = -0.5
            keep_forward = 1.0 # 1.0
            reach_grasp_area = 1000.0
            # lin_vel_y = -0.1

            stand_still = 50.0 # 1.0
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
        entropy_coef = 0.003
        # entropy_coef = 0.05
        
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'go2_nav_flat'

        save_interval = 200  # save model every n iterations
        max_iterations = 5000  # maximum number of training iterations
        
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
