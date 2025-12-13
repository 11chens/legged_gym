
import torch

def test_obs_logic():
    num_envs = 10
    num_props = 31
    history_len = 5
    nav_history_len = 10
    num_nav_commands = 15
    
    # Mock buffers
    sigma_points_body = torch.zeros(num_envs, 5, 3)
    obs_hist_buffer = torch.zeros(num_envs, history_len, num_props)
    delay_nav_commands_hist_buffer = torch.zeros(num_envs, nav_history_len, num_nav_commands)
    
    # Mock views
    v_sigma = sigma_points_body.view(num_envs, -1)
    v_hist = obs_hist_buffer.view(num_envs, -1)
    v_delay = delay_nav_commands_hist_buffer.view(num_envs, -1)
    
    print(f"sigma: {v_sigma.shape}")
    print(f"hist: {v_hist.shape}")
    print(f"delay: {v_delay.shape}")
    
    obs_buf = torch.cat([v_sigma, v_hist, v_delay], dim=-1)
    print(f"obs_buf: {obs_buf.shape}")
    
    expected = 15 + 155 + 150
    print(f"Expected: {expected}")
    
    if obs_buf.shape[1] != expected:
        print("MISMATCH!")
    else:
        print("MATCH!")

if __name__ == "__main__":
    test_obs_logic()
