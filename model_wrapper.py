import torch
import torch.nn as nn
import numpy as np
import math

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class DiffusionModelWrapper(nn.Module):
    def __init__(self, model, diffusion_steps=4000):
        super().__init__()
        self.model = model
        self.diffusion_steps = diffusion_steps
        
        # Match original training config
        self.image_size = 256
        self.num_channels = 64
        self.learn_sigma = True  # 6 output channels
        self.attention_resolutions = (16, 8)
        self.noise_schedule = 'cosine'
        
        # Input normalization params
        self.register_buffer('mean', torch.tensor([0.5]).reshape(1, 1, 1, 1))
        self.register_buffer('std', torch.tensor([0.5]).reshape(1, 1, 1, 1))

    def normalize_input(self, x):
        """Normalize input to [-1, 1] range"""
        x = x.float()
        x = (x - self.mean) / self.std
        return x

    def forward(self, x, t):
        """
        Forward pass with input normalization
        :param x: Input tensor [B, C, H, W]
        :param t: Timestep tensor [B]
        """
        # Handle input normalization
        x = self.normalize_input(x)
        
        # Use the model's original timestep processing
        # Pass timesteps directly to the model
        out = self.model(x, t)
        
        return out

    @torch.no_grad()
    def trace(self, example_inputs):
        """
        Method for tracing the model with proper input handling
        """
        self.eval()
        
        # Prepare example inputs - normalize input but pass timestep directly
        x, t = example_inputs
        x = self.normalize_input(x)
        
        # Return traced version
        return torch.jit.trace(self, (x, t))
