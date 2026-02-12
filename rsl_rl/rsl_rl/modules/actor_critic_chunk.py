import torch
import torch.nn as nn
from torch.distributions import Normal
from .actor_critic_encoder import ActorCriticEncoder

class ActorCriticChunk(ActorCriticEncoder):
    """
    MLP-Encoder based Policy with Action Chunking (ACT).
    Inherits from ActorCriticEncoder but modifies the Output Head and Training Logic.
    """
    def __init__(self, num_actions, chunk_size=10, visual_dropout_prob=0.0, init_noise_std=0.5, **kwargs):
        self.chunk_size = chunk_size
        self.visual_dropout_prob = visual_dropout_prob
        
        print(f"ActorCriticChunk Init: Input Noise Std={init_noise_std}")
        
        super().__init__(num_actions, init_noise_std=init_noise_std, **kwargs)
        
        # Modify Actor Head (which is self.actor.actor in the Policy object)
        # self.actor is the Policy instance.
        # self.actor.actor is the MLP Sequential.
        
        # Access the last linear layer
        last_layer = self.actor.actor[-1]
        in_features = last_layer.in_features
        
        # Rebuild last layer to output Chunk Size * Num Actions
        # We recreate the sequential block to ensure we replace the head correctly
        new_layers = list(self.actor.actor.children())[:-1]
        new_layers.append(nn.Linear(in_features, num_actions * chunk_size))
        self.actor.actor = nn.Sequential(*new_layers)
        
        # NOTE: Removed explicit small-weight initialization to match ActorCriticEncoder baseline behavior.
        # This helps initial exploration when chunk_size=1.
        
        print(f"ActorCriticChunk: Chunk Size={chunk_size}, Output Dim={num_actions * chunk_size}, Dropout={visual_dropout_prob}")

    def act(self, observations, **kwargs):
        # 1. Split Privileged Information
        priv_info, observations = self.get_priv_separated(observations)
        
        # 2. Extract components using the internal Policy helper
        # prop_hist: Full interleaved history [Batch, Hist*Props]
        # ext_hist: Flattened External/Nav history [Batch, Hist*Nav]
        # curr_obs: Last step [Batch, Props]
        prop_hist, ext_hist, curr_obs = self.actor.extract_obs(observations)

        # 3. Visual Dropout on External Features (Nav/Sigma)
        # Effectively simulates entering a blind spot where visual/nav data is lost (-1)
        # if self.training and self.visual_dropout_prob > 0:
        #     # Mask generation: 1 = Keep, 0 = Blind
        #     # Single mask per env
        #     dropout_mask = (torch.rand(observations.shape[0], 1, device=observations.device) > self.visual_dropout_prob).float()
            
        #     # Apply dropout: Replace with -1.0 if blind
        #     ext_hist = ext_hist * dropout_mask + (1.0 - dropout_mask) * (-1.0)
            
        # 4. Forward Pass
        if self.actor.encoder is not None:
             # Pass modified ext_hist
             latent = self.actor.encoder(prop_hist, ext_hist)
             actor_obs = latent
        else:
             # Fallback for no encoder
             actor_obs = prop_hist
        
        # 5. Actor Head
        mean = self.actor.actor(actor_obs)
        
        # 6. Distribution
        # We need to broadcast the std dev to the chunk size
        # self.std is [num_actions] parameter, we repeat it to [num_actions * chunk_size]
        self.distribution = Normal(mean, mean*0. + self.std.repeat(self.chunk_size))
        
        return self.distribution.sample()

