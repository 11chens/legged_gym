
import isaacgym
import torch
from legged_gym.envs.go2.go2_nav_config import Go2NavFlatCfg
from legged_gym.envs.base.legged_robot_nav import LeggedRobotNav

def test_shapes():
    cfg = Go2NavFlatCfg()
    # Mock simulation parameters (minimal needed for init)
    sim_params = {"sim": {"dt": 0.02, "up_axis": 1, "use_gpu_pipeline": True, "gravity": [0, 0, -9.81], "physx": {"num_threads": 1, "solver_type": 1, "use_gpu": True, "num_subscenes": 1, "max_gpu_contact_pairs": 1024*1024}}}
    
    # We can't easily instantiate LeggedRobotNav without a running Isaac Gym instance.
    # Isaac Gym requires a window or headless mode setup which might be complex here.
    # However, we can inspect the Config object directly to verify our calculations.
    
    print(f"NUM_NAV_COMMANDS: {cfg.env.num_nav_commands}")
    print(f"NUM_PROPS: {cfg.env.num_props}")
    print(f"NUM_OBSERVATIONS: {cfg.env.num_observations}")
    print(f"HISTORY_LEN: {cfg.env.history_len}")
    print(f"NAV_HISTORY_LEN: {cfg.env.nav_history_len}")
    print(f"NUM_SIGMA_POINTS: {cfg.env.num_sigma_points}")
    print(f"NUM_PRIV: {cfg.env.num_priv}")
    
    # Calculate expected size
    num_nav_commands = cfg.env.num_nav_commands
    num_props = cfg.env.num_props
    history_len = cfg.env.history_len
    nav_history_len = cfg.env.nav_history_len
    num_priv = cfg.env.num_priv
    
    calculated_obs = num_nav_commands * nav_history_len + num_props * history_len + num_priv
    print(f"Calculated num_observations: {calculated_obs}")
    
    # Check if num_props matches components
    # num_props = num_nav_actions + num_nav_commands + 12
    num_nav_actions = cfg.env.num_nav_actions
    calculated_props = num_nav_actions + num_nav_commands + 12
    print(f"Calculated num_props: {calculated_props}")

if __name__ == "__main__":
    test_shapes()
