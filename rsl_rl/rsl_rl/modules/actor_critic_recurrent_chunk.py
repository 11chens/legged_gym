
import torch
import torch.nn as nn
from torch.distributions import Normal
from .actor_critic_recurrent_light import ActorCriticRecurrentLight
from rsl_rl.utils import unpad_trajectories

class ActorCriticRecurrentChunk(ActorCriticRecurrentLight):
    def __init__(self, num_actions, chunk_size=10, visual_dropout_prob=0.1, **kwargs):
        # We must initialize self parameters before super just in case, or capture them
        self.chunk_size = chunk_size
        self.visual_dropout_prob = visual_dropout_prob
        
        super().__init__(num_actions, **kwargs)
        
        # Redefine Actor Head for Chunking
        # Get the input dim of the last layer of the existing head
        last_layer = self.actor_head[-1]
        in_features = last_layer.in_features
        
        # Rebuild head with new output dimension
        new_layers = list(self.actor_head.children())[:-1]
        new_layers.append(nn.Linear(in_features, num_actions * chunk_size))
        self.actor_head = nn.Sequential(*new_layers)
        
        print(f"ActorCriticRecurrentChunk: Chunk Size={chunk_size}, Output Dim={num_actions * chunk_size}, Dropout={visual_dropout_prob}")

    def act(self, observations, masks=None, hidden_states=None):
        # 1. Visual Dropout (Blind Spot Simulation)
        if self.training and self.visual_dropout_prob > 0:
            # Create a mask for the batch: [Batch, 1]
            dropout_mask = (torch.rand(observations.shape[0], 1, device=observations.device) > self.visual_dropout_prob).float()
            
            # Identify indices to dropout (Nav/Visual part)
            # Based on RecurrentLight: Nav is [9 : 9+num_nav_commands]
            # Since observations is history interleaved [Batch, Hist*Props], 
            # We need to mask out Nav features across ALL history steps ?? 
            # RecurrentLight extract_obs only takes the LAST frame.
            # So masking the input `observations` blindly might be complex if interleaved.
            # However `extract_obs` logic is:
            # curr_obs = obs_seq[..., -1, :]
            # nav_obs = curr_obs[..., start:end]
            
            # So we can just act on the result of extract_obs?
            # But we need to reimplement act to intervene.
            pass

        # === Derived from ActorCriticRecurrentLight.act ===
        
        # 1. Split Privileged
        priv_info, observations = self.get_priv_separated(observations)
        
        # 2. Extract Latest Frame & Encode
        rnn_input_data, body_obs = self.extract_obs(observations)
        
        # Apply Dropout HERE on rnn_input_data (which is nav_obs)
        if self.training and self.visual_dropout_prob > 0:
             dropout_mask = (torch.rand(rnn_input_data.shape[0], 1, device=rnn_input_data.device) > self.visual_dropout_prob).float()
             # If dropout_mask is 0 (blind), input is -1
             # We interpolate: mask * input + (1-mask) * (-1)
             rnn_input_data = rnn_input_data * dropout_mask + (1.0 - dropout_mask) * (-1.0)
        
        latent = self._encode(rnn_input_data.detach())
        
        # 3. RNN
        rnn_input = latent
        h_a = hidden_states
        
        if self.shared_memory:
            rnn_out = self.memory_unify(rnn_input, masks, h_a)
        else:
            rnn_out = self.memory_a(rnn_input, masks, h_a)

        if not (masks is not None): 
             # Inference mode
             rnn_out = rnn_out.squeeze(0)
        
        # 4. Concatenate Body Obs if needed
        if self.rnn_nav_only and body_obs is not None:
             if masks is not None:
                 body_obs = unpad_trajectories(body_obs, masks)
             head_input = torch.cat([rnn_out, body_obs], dim=-1)
        else:
             head_input = rnn_out

        # 5. Actor Head
        action_mean = self.actor_head(head_input)
        
        # Output shape is [Batch, Chunk*Actions]. 
        # We don't sample from a diagonal Gaussian for the whole chunk usually in ACT?
        # User Doc: "Policy Output: A_t ... prediction"
        # Usually ACT uses deterministic output or VAE.
        # But here we are integrating into PPO.
        # PPO requires log_prob.
        # We process the Chunk as a single high-dim action vector.
        
        self.distribution = Normal(action_mean, action_mean*0. + self.std.repeat(self.chunk_size))
        
        # Return the sampled chunk
        return self.distribution.sample()

    # Need to update std parameter too in init if it's used?
    # Base class init creates self.std of size num_actions.
    # We shouldn't change self.std shape if we want to load old weights?
    # No, this is a new architecture (Output dim changed). Weights won't match anyway for the head.
