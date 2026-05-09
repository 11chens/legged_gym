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

NUM_SIGMA_POINTS = (2 * 3 + 1)
NUM_NAV_COMMANDS = NUM_SIGMA_POINTS * 3 # all sigma points (3D)
EPISODE_LENGTH_S = 6

class Go2NavFlatCfg( LeggedRobotNavCfg ):
    target = TargetCfg
    debug_viz = True
    class loco:
        use_loco_policy = True
        loco_body_model_file = 'controller/model_light.jit'
        loco_obs_dim = 47
        loco_history_len = 5

    class env(LeggedRobotNavCfg.env):
        num_position = 3 # x, y, z
        num_nav_actions = 4 # vx, vy, vyaw, pitch
        nav_history_len = 10
        history_len = 5
        num_sigma_points = NUM_SIGMA_POINTS
        num_nav_commands = NUM_NAV_COMMANDS
        num_task_flags = 1
        num_props = num_nav_actions + num_nav_commands + num_task_flags + 9 # (lin_vel(3)), ang_vel(3), gravity(3)
        num_priv = 1 + NUM_NAV_COMMANDS # task_id + nav commands
        num_observations = num_props * history_len + num_priv + num_nav_commands * nav_history_len
        num_privileged_obs = None

        num_envs = 2048
        episode_length_s = EPISODE_LENGTH_S # episode length in seconds  # will be randomized in [s-minus, s]
        debug_viz = False

        # Optimal Grasp Pose Parameters
        grasp_offset_long = -0.07 # [m] Distance from long-axis vertex (Head/Tail)
        grasp_offset_short = -0.05 # [m] Distance from object surface (Short axis)
        grasp_offset_place = -0.1 # [m] Distance from object surface (Short axis)
        
        # Hint Pose Parameters
        hint_dist_long = 0.4 # [m] Distance from long-axis vertex for hint pose
        hint_dist_short = 0.4 # [m] Distance from robot start point for hint pose
        hint_dist_place = 0.4 # [m] Distance from box center for hint pose in Place task

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
        resampling_time = 2.0
        enable_delay = True
        max_delay_time_ms = 50  # maximum delay time in milliseconds
        min_delay_time_ms = 0   # minimum delay time in milliseconds
        # navigation refresh interval range [min_steps, max_steps]
        # e.g., [1, 2] means 50Hz to 25Hz if dt=20ms, [2, 2] means fixed 25Hz, [1, 3] means 50Hz to ~16.7Hz
        nav_refresh_steps_range = [1, 2] 
        cmds_alpha_range = [0.1, 0.55] # alpha range for smoothing the commands, sampled from [0.1, 0.5] with step 0.05
        cmds_alpha_step = 0.05
        # invalid commands
        enable_invalid_cmds = True
        enable_out_of_view_drift = True # if True, add random walk drift when out of view
        enable_drift_curriculum = True
        drift_scale = 0.01 # scale of the random walk drift per step
        max_drift = 0.1 # [m] maximum drift distance
        
        # Adaptive Drift Curriculum
        # as success_ratio -> 1.0 (or target), drift_scale -> drift_scale_range[1]
        drift_scale_range = [0.005, 0.02]
        max_drift_range = [0.05, 0.1]
        drift_curriculum_threshold = 0.1 # success ratio reference for max difficulty (as per instruction)
        
        # max_out_of_view_duration = 2.0 # [s] the duration to keep the out of view coordinates

        enable_place_offset = False # If True, apply z-offset when placing, farther away from the object
        place_offset_value = 0.2 # [m] z-offset when placing
        place_offset_error_margin = 0.05 # [m] error margin for place offset

        frame_drop_prob = 0.05 # Probability of dropping a frame (simulating sensor failure)
        hold_time_s = 1.0 # [s] time to hold to consider as success
        place_pitch_target = -0.2 # [rad] target pitch angle when placing

        class ranges:
            limit_vx = [-0.5, 0.5]  # [m/s]
            limit_vy = [-0.5, 0.5]  # [m/s]
            limit_vyaw = [-0.5, 0.5]  # [rad/s]
            limit_pitch = [-3.14/6, 3.14/6]  # [rad]
            heading = [-0.3, 0.3]  # a residual heading plus theta
    
        class resample:
            adjust_obj_pose = True
            force_look_upwards = True
            enable_success_feeding = True # If True, use success  feeding mechanism
            force_lookup_prob = 0.2 # Probability of forcing a look-up when adjusting object pose
            min_steps = 100 # Minimum steps before resampling on the way
            look_up_duration = 0.5 # [s] Duration to maintain look-up pitch
            resample_interval_steps = 150 # Resample every N steps
            # Feeding object near target
            feeding_prob = 0.33 # Probability of feeding when using success feeding
            # Replay failed scenarios
            replay_failed_prob = 0.7 # Probability of replaying a failed scenario
            
            # Adaptive Curriculum Parameters
            curriculum_threshold = 0.1 # Success ratio max reference
            feeding_prob_range = [0.2, 0.5] # [min, max] probability
            replay_failed_prob_range = [0.2, 0.8] # [min, max] probability

            class ranges:
                # dist_fwd = [0.1, 0.5] # min max [m]
                dist_fwd = [0.0, 0.3] # min max [m]
                dist_lat = [-0.2, 0.2] # min max [m]
                min_dist = 0.5  # minimum time to reach the target [s]
                max_dist = 1.5  # maximum time to reach the target [s]

    class gripper:
        gripper_offset = [0.41, 0.0, -0.07] # [m] offset from base link to gripper center in base frame
        gripper_width = 0.12 # [m]
    
    class camera_sensor:
        # visualization options
        enable_camera = False  # if True, the image of isaacgym is enabled, otherwise it is disabled
        num_vis_points = 100  # number of points sampling from target points for visualization
        vis_gripper_position = True # if True, visualize the gripper position in the world
        vis_sigma_axes = False  # if True, visualize the sigma points axes in the world
        vis_object_axes = False  # if True, visualize the object coordinate axes in the world
        vis_head_tail_points = False  # if True, visualize the head and tail points in the world
        vis_target_points_in_image = True # if True, visualize target points (green) in image
        vis_sigma_3d_in_image = True # if True, visualize 3D sigma points (blue) in image
        vis_sigma_3d_in_world = False # if True, visualize 3D sigma points (white) in world
        vis_sigma_2d = False # if True, visualize 2D sigma points (yellow)
        vis_sigma_y_spread = False # if True, visualize the sigma points Y-spread axis (yellow)
        save_debug_images = False # if True, save debug images to disk, otherwise view in real-time

        # Environment configuration
        clip_invalid = False # if True, the invalid image coordinates are clipped to -1, otherwise they are kept as is

        # Camera extrinsics and intrinsics
        fix_extrinsics = False  # if True, the camera extrinsics are fixed, otherwise they are randomized
        fix_intrinsics = True  # if True, the camera intrinsics are fixed, otherwise they are randomized
        fix_img_shape = True  # if True, the image shape is fixed, otherwise it is randomized

        class intrinsics: # Intrinsics parameters
            horizontal_fov = 70.26
            img_width = 320
            img_height = 180
            fx = 227.3841552734375
            fy = 227.24505615234375
            cx = 162.8849639892578
            cy = 92.34309387207031

            horizontal_fov_range = [-2.0, 2.0] # [degree]
            img_height_range = [90, 720] # [pixel]
            img_width_range = [160, 1280] # [pixel]

        class extrinsics: # Extrinsics parameters
            #  ================= fixed extrinsics =================
            translation = [0.305, 0.017, 0.128]  # Translation: forward, left, upward
            angles = [0.0, 33.0, 0.0]  # Euler angles: yaw, pitch, roll
            
            #  ================= random extrinsics =================
            # Randomization ranges around the fixed extrinsics
            yaw_range = [-0.5, 0.5]   # [degree]
            pitch_range = [-2.0, 2.0] # [degree]
            roll_range = [-0.5, 0.5]  # [degree]

            dx_range = [-0.02, 0.02]   # [m]
            dy_range = [-0.005, 0.005]   # [m]
            dz_range = [-0.02, 0.02]   # [m]

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
        friction_range = [0.2, 5.0]
        randomize_restitution = True
        restitution_range = [0.0, 1.0]
        randomize_base_mass = True
        added_mass_range = [-1., 2.0]
        randomize_base_com = True
        added_com_range = [-0.05, 0.05]
        push_robots = False
        push_interval_s = 5
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
        noise_level = 1.0
        class noise_scales:
            dof_pos = 0.01 # 0.01 
            dof_vel = 1.0 # 1.0
            lin_vel = 0.1 # 0.1
            ang_vel = 0.1 # 0.2
            gravity = 0.1 # 0.05
            pitch = 0.1 # 0.1
            euler_rpy = 0.1 # 0.1
            
            # Parametric Sigma Point Noise (Sim2Real)
            nav_pos_3d = [0.0, 0.0, 0.1] # Center noise [m] in Camera Frame (X, Y, Z)
            nav_scale_3d = 0.1 # Scale noise (proportional)
            nav_rot_3d = 0.1 # Rotation noise [rad] (~5.7 deg)
            
            nav_pos_2d = 0.005 # Center noise [normalized]
            nav_scale_2d = 0.05 # Scale noise (proportional)
            nav_rot_2d = 0.1 # Rotation noise [rad] (~6 deg)

    class rewards():
        class opt_pos_weights():
            x = 2.0
            y = 1.0
            z = 0.1

        class opt_rot_weights():
            yaw = 1.0
            pitch = 0.5
            roll = 0.0

        class scales():
            ang_vel_xy = -0.1
            orientation_y = -2.0
            nav_action_rate = -0.01
            nav_action_limit = -0.1

            # New rewards for sigma points
            missing_sigma_points = -0.1
            successful_grasp = 20.0
            backup = -4
            sequential_reaching = 0.3

        soft_dof_pos_limit = 0.95
        base_height_target = 0.25
        only_positive_rewards = False
        position_target_sigma_soft = 2.0
        position_target_sigma_tight = 0.5
        heading_target_sigma = 1.0
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.85
        max_contact_force = 100.
        tracking_sigma = 0.1
        weight_track_pick_long_pos = 3.0
        weight_track_pick_short_pos = 5.0
        weight_track_place_pos = 5.0
        rot_track_sigma = 0.04
        pos_track_sigma = 0.02
        vel_track_sigma = 0.04
        soft_sigma = 0.04
        stand_still_lin_sigma = 0.04
        stand_still_ang_sigma = 0.08

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
        
        # policy_class_name = 'ActorCriticEncoder' # good performance
        policy_class_name = 'ActorCriticTCN'
        # policy_class_name = "ActorCriticRecurrentEncoder" # bad performance
        # policy_class_name = "ActorCriticRecurrentLight" # bad performance
        # policy_class_name = "ActorCriticChunk" 
        algorithm_class_name = 'PPO'
        # algorithm_class_name = 'PPOChunk'

    class policy( LeggedRobotCfgPPO.policy ):
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        rnn_type = 'gru'
        chunk_size = 10  # Number of steps to process in a chunk
        
        # Optimized TCN params for speed
        # tcn_channels = [32, 32, 32] # Default was [64, 64, 64, 64]
        # tcn_kernel_size = 3 # Default was 5

        # Optimized TCN params for speed
        tcn_channels = [32, 32, 32] 
        tcn_kernel_size = 5 
        tcn_dropout = 0.1

        # init_noise_std = 0.3


        # actor_hidden_dims = [256, 128, 64]
        # critic_hidden_dims = [256, 128, 64]

        # actor_hidden_dims = [128, 64, 32]
        # critic_hidden_dims = [128, 64, 32]
        has_encoder = True
        # rnn_hidden_size = 128

        # actor_hidden_dims = [128, 64, 32]
        # critic_hidden_dims = [128, 64, 32]
