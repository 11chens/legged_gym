import torch

class ActionIntegrator:
    def __init__(self, num_envs, chunk_size, action_dim, device='cpu'):
        self.num_envs = num_envs
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.device = device
        
        # Buffer to store predicted action chunks
        # We need to store history to aggregate
        # Dimensions: [num_envs, chunk_size (past steps), chunk_size (prediction horizon), action_dim]
        # However, a simpler way is:
        # At time t:
        # We have predictions made at t, t-1, ..., t-(k-1)
        # Prediction made at t-j is for [t-j, ..., t-j+k-1]
        # The j-th element of that prediction (0-indexed) corresponds to time (t-j) + j = t.
        # So we want: pred_t[0], pred_{t-1}[1], ..., pred_{t-(k-1)}[k-1]
        
        # We can implement a rolling buffer of shape [num_envs, chunk_size, chunk_size, action_dim]
        # But efficiently:
        # We just need to store the relevant diagonals.
        # Actually, simpler: Store the last k chunks.
        # Buffer: [num_envs, chunk_size, chunk_size, action_dim]
        # dim 1 acts as a ring buffer for "time since prediction"
        
        self.action_history = torch.zeros(num_envs, chunk_size, chunk_size, action_dim, device=device)
        self.step_idx = 0
        self.valid_history_len = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.action_history.fill_(0)
            self.step_idx = 0
            self.valid_history_len.fill_(0)
        else:
            self.action_history[env_ids] = 0.0
            self.valid_history_len[env_ids] = 0

    def add_and_aggregate(self, new_chunk):
        """
        new_chunk: [num_envs, chunk_size * action_dim] or [num_envs, chunk_size, action_dim]
        """
        if new_chunk.ndim == 2:
            new_chunk = new_chunk.view(self.num_envs, self.chunk_size, self.action_dim)
            
        # 1. Update Buffer
        # We use a ring buffer index for the "past steps" dimension
        insert_idx = self.step_idx % self.chunk_size
        self.action_history[:, insert_idx, :, :] = new_chunk
        
        # Increment valid history length for all envs (saturated at chunk_size)
        self.valid_history_len = torch.clamp(self.valid_history_len + 1, max=self.chunk_size)
        
        # 2. Aggregate
        aggregated_action = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        
        # We perform a weighted sum where invalid history has 0 weight
        # But for efficiency, we can iterate and add only where valid
        # Vectorized approach:
        
        for j in range(self.chunk_size):
            # Access past prediction stored at ring buffer index
            past_idx = (insert_idx - j) % self.chunk_size
            
            # Check which envs have this history valid
            # j=0 (current) represents history len 1.
            # j=k-1 represents history len k.
            # So history is valid if valid_history_len > j
            mask = (self.valid_history_len > j).float().unsqueeze(-1)
            
            component = self.action_history[:, past_idx, j, :]
            aggregated_action += component * mask
            
        # Divide by valid count (avoid division by zero if something weird happens, though +1 above ensures min 1)
        divisor = self.valid_history_len.float().unsqueeze(-1)
        aggregated_action /= divisor
        
        self.step_idx += 1
        return aggregated_action
