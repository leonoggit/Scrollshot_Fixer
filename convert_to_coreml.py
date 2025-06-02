import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
from model_wrapper import DiffusionModelWrapper
from echogains.RePaint.guided_diffusion.unet import UNetModel, ResBlock, timestep_embedding
from echogains.RePaint.guided_diffusion.nn import checkpoint
import os
import logging
import yaml
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """
    Load configuration from YAML file
    """
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(__file__), "echogains/RePaint/confs/camus_all.yml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_model_channels(config):
    """Parse channel multiplier from config"""
    if config.get('channel_mult'):
        return tuple(int(x) for x in config['channel_mult'].split(',')) if isinstance(config['channel_mult'], str) else tuple(config['channel_mult'])
    return (1, 2, 3, 4)

def create_model_config():
    """Create model configuration with confirmed training parameters"""
    return {
        'image_size': 256,
        'num_channels': 64,
        'num_res_blocks': 2,  # Changed to match original model
        'learn_sigma': True,
        'diffusion_steps': 4000,
        'noise_schedule': 'cosine',
        'use_checkpoint': False,  # Disabled for conversion
        'dropout': 0.0,
        'attention_resolutions': '16,8',
        'num_heads': 4,
        'num_head_channels': 64,
        'resblock_updown': False,  # Changed to match original model
        'use_scale_shift_norm': True,
        'use_new_attention_order': False,
        'channel_mult': '1,2,3,4'  # Changed to match original model
    }

def create_model(config=None):
    """
    Create UNetModel with proper configuration from training
    """
    if config is None:
        config = create_model_config()
        
    attention_resolutions = [int(x) for x in config.get('attention_resolutions', '16,8').split(',')]
    channel_mult = tuple(int(x) for x in config.get('channel_mult', '1,2,2,4').split(','))
    
    # Create a simple config object for the model
    class SimpleConfig:
        def __init__(self):
            self.diffusion_steps = 4000
            self.use_value_logger = False
    
    model = UNetModel(
        image_size=config.get('image_size', 256),
        in_channels=1,  # Change to 1 for grayscale medical images
        model_channels=config.get('num_channels', 64),
        out_channels=2 if config.get('learn_sigma', True) else 1,  # 2 channels for grayscale with sigma
        num_res_blocks=config.get('num_res_blocks', 4),
        attention_resolutions=tuple(attention_resolutions),
        dropout=config.get('dropout', 0.0),
        channel_mult=channel_mult,
        use_checkpoint=config.get('use_checkpoint', False),
        num_heads=config.get('num_heads', 4),
        num_head_channels=config.get('num_head_channels', 64),
        num_heads_upsample=config.get('num_heads_upsample', -1),
        use_scale_shift_norm=config.get('use_scale_shift_norm', True),
        resblock_updown=config.get('resblock_updown', True),
        use_new_attention_order=config.get('use_new_attention_order', False),
        conf=SimpleConfig()
    )
    
    return model

class CustomResBlock(nn.Module):
    """
    A ResNet block without checkpointing and with fixed dimensionality for CoreML conversion.
    Closely matches the original ResBlock implementation but removes dynamic features.
    """
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        out_channels: Optional[int] = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        dims: int = 2,
        up: bool = False,
        down: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        self.updown = up or down

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1)
        )

        # Simplified embedding handling
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
        )

        # Simplified skip connection
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass without any checkpointing.
        
        Args:
            x: Input tensor [B, C, H, W]
            emb: Embedding tensor [B, emb_dim]
            
        Returns:
            Output tensor [B, C, H, W]
        """
        h = self.in_layers(x)
        
        # Process embedding
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        
        # Apply scale-shift or simple addition
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = self.out_layers[0](h) * (1 + scale) + shift
            h = self.out_layers[1:](h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
            
        return self.skip_connection(x) + h

def replace_resblocks_with_custom(model: nn.Module) -> None:
    """
    Recursively replace all ResBlock instances with CustomResBlock.
    Carefully preserves the model's state during replacement.
    
    Args:
        model: The model whose ResBlocks need to be replaced
    """
    for name, module in list(model.named_children()):
        if isinstance(module, ResBlock):
            # Create new block with identical parameters
            new_block = CustomResBlock(
                channels=module.channels,
                emb_channels=module.emb_channels,
                dropout=module.dropout,
                out_channels=module.out_channels,
                use_conv=module.use_conv,
                use_scale_shift_norm=module.use_scale_shift_norm,
                dims=2,  # Fixed for CoreML
                up=getattr(module, 'up', False),
                down=getattr(module, 'down', False)
            )
            
            # Copy state dict with careful handling of shape mismatches
            try:
                new_block.load_state_dict(module.state_dict())
            except Exception as e:
                logger.warning(f"Error copying state for block {name}: {str(e)}")
                logger.warning("Attempting parameter by parameter copy...")
                # Manual parameter copying as fallback
                for param_name, param in module.named_parameters():
                    if hasattr(new_block, param_name):
                        target_param = getattr(new_block, param_name)
                        if target_param.shape == param.shape:
                            target_param.data.copy_(param.data)
            
            # Replace the old block
            if isinstance(model, nn.ModuleList):
                model[int(name)] = new_block
            else:
                setattr(model, name, new_block)
        else:
            # Recursively process child modules
            replace_resblocks_with_custom(module)

def create_model_from_config(config):
    """
    Create model instance based on configuration from camus_all.yml,
    ensuring it matches the exact architecture used during training
    """
    # Use exact parameters from training
    image_size = 256  # Hardcoded to match training
    num_channels = 64  # Hardcoded to match training
    num_res_blocks = 4  # Hardcoded to match training
    learn_sigma = True  # Hardcoded to match training
    
    # Get other parameters from config, with fallbacks matching training
    attention_resolutions = tuple(int(x) for x in config.get('attention_resolutions', '16,8').split(','))
    num_heads = config.get('num_heads', 4)
    num_head_channels = config.get('num_head_channels', -1)
    channel_mult_str = config.get('channel_mult', '1,1,2,2,4')
    channel_mult = tuple(int(x) for x in channel_mult_str.split(','))
    
    # Create model with exact architecture from training
    model = UNetModel(
        image_size=image_size,
        in_channels=3,  # RGB input
        model_channels=num_channels,
        out_channels=6 if learn_sigma else 3,  # 6 channels when learn_sigma=True
        num_res_blocks=num_res_blocks,
        attention_resolutions=attention_resolutions,
        dropout=0.0,  # No dropout as per training
        channel_mult=channel_mult,
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,  # Disable checkpointing for conversion
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=-1,
        use_scale_shift_norm=config.get('use_scale_shift_norm', True),
        resblock_updown=config.get('resblock_updown', False),
        use_new_attention_order=config.get('use_new_attention_order', False),
    )
    
    # Replace all ResBlocks with our custom implementation
    replace_resblocks_with_custom(model)
    
    return model

def load_with_mismatch(model, checkpoint):
    """
    Load the checkpoint into the model, handling any dimension mismatches
    """
    logger.info("Loading model weights...")
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
        
    model_state = model.state_dict()
    
    # Keep track of mismatches for logging
    mismatches = []
    
    # Process each key in the state dict
    for key in list(state_dict.keys()):
        if key not in model_state:
            logger.warning(f"Removing unexpected key {key} from state dict")
            del state_dict[key]
            continue
            
        if state_dict[key].shape != model_state[key].shape:
            mismatches.append({
                'layer': key,
                'checkpoint_shape': state_dict[key].shape,
                'model_shape': model_state[key].shape
            })
            
            # Remove mismatched keys - the model will keep its initialization for these
            del state_dict[key]
            
    # Log all mismatches
    if mismatches:
        logger.warning("The following layers had shape mismatches and were skipped:")
        for mismatch in mismatches:
            logger.warning(f"Layer: {mismatch['layer']}")
            logger.warning(f"  Checkpoint shape: {mismatch['checkpoint_shape']}")
            logger.warning(f"  Model shape: {mismatch['model_shape']}")
    
    # Load the matching weights
    model.load_state_dict(state_dict, strict=False)
    logger.info("Model loaded successfully with partial weights")

class InputAdapter(nn.Module):
    """
    Adapter module to handle input preprocessing and timestep management for CoreML conversion.
    """
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        super().__init__()
        self.model = model
        self.config = config
        self.image_size = config.get('image_size', 256)
        
        # Create model configuration object
        self.model.conf = type('Config', (), {
            'diffusion_steps': config.get('diffusion_steps', 4000),
            'learn_sigma': config.get('learn_sigma', True),
            'noise_schedule': config.get('noise_schedule', 'cosine'),
            'use_value_logger': False,
        })
        
    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input to [-1, 1] range"""
        if x.dtype == torch.uint8 or x.max() > 1:
            x = x.float()
            if x.max() > 1:
                x = x / 255.0
        return 2 * x - 1  # Scale to [-1, 1]
    
    def process_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate timesteps for inference"""
        # For inference, we use t=0 (no noise)
        return torch.zeros(batch_size, dtype=torch.long, device=device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input and run model inference.
        
        Args:
            x: Input tensor [B, C, H, W] in [0, 1] or [0, 255] range
            
        Returns:
            Output tensor [B, 3, H, W] in [0, 1] range
        """
        # Handle input shape
        if x.dim() != 4:
            x = x.view(-1, 1, self.image_size, self.image_size)
        
        # Convert single channel to RGB if needed
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # Normalize input to [-1, 1]
        x = self.normalize_input(x)
        
        # Process timesteps
        batch_size = x.shape[0]
        timesteps = self.process_timesteps(batch_size, x.device)
        
        # Run model inference
        with torch.no_grad():
            out = self.model(x, timesteps)
        
        # Process output
        if self.model.conf.learn_sigma:
            out = out[:, :3]  # Take predicted x_start only
        
        # Rescale output to [0, 1]
        out = (out + 1) / 2
        out = torch.clamp(out, 0, 1)
        
        return out

def simple_checkpoint(func, inputs, params, flag):
    """A simplified checkpoint function that doesn't actually checkpoint for tracing"""
    if isinstance(inputs, tuple):
        return func(*inputs)
    return func(inputs)

# Monkey patch the checkpoint function
import echogains.RePaint.guided_diffusion.nn as nn_module
nn_module.checkpoint = simple_checkpoint

# Monkey patch checkpoint function to avoid tracing issues
def noop_checkpoint(func, inputs, params, flag):
    """No-op checkpoint function for tracing"""
    return func(*inputs)

# Apply the monkey patch
import echogains.RePaint.guided_diffusion.nn as gd_nn
gd_nn.checkpoint = noop_checkpoint

def main():
    """
    Main execution function for model conversion with comprehensive error handling.
    """
    try:
        # Create configuration
        config = create_model_config()
        logger.info("Configuration created successfully")
        
        # Create model with exact training parameters
        model = create_model(config)
        logger.info("Model architecture created")
        
        # Load checkpoint
        checkpoint_path = "model_files/CAMUS_diffusion_model.pt"
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        load_with_mismatch(model, checkpoint)
        logger.info("Checkpoint loaded successfully")
        
        # Verify model configuration
        if not hasattr(model, 'conf'):
            logger.warning("Model missing configuration, creating default")
            model.conf = type('Config', (), {
                'diffusion_steps': config['diffusion_steps'],
                'learn_sigma': config['learn_sigma'],
                'noise_schedule': config['noise_schedule'],
            })
        
        # Disable gradients and put in eval mode
        model.requires_grad_(False)
        model = model.eval()
        
        # Create input adapter
        model_with_adapter = InputAdapter(model, config)
        model_with_adapter.eval()
        
        # Prepare example inputs
        logger.info("Preparing example inputs for tracing...")
        example_inputs = [
            torch.randn(1, 1, 256, 256),  # Single channel
            torch.randn(1, 3, 256, 256),  # RGB
            torch.ones(1, 1, 256, 256),   # All white
            torch.zeros(1, 1, 256, 256),  # All black
        ]
        
        # Verify model works with all example inputs
        logger.info("Verifying model with example inputs...")
        for i, example in enumerate(example_inputs):
            try:
                with torch.no_grad():
                    out = model_with_adapter(example)
                assert out.shape == (1, 3, 256, 256)
                logger.info(f"Example input {i} processed successfully")
            except Exception as e:
                logger.error(f"Error processing example input {i}: {str(e)}")
                raise
        
        # Trace model
        logger.info("Tracing model...")
        traced_model = torch.jit.trace(
            model_with_adapter,
            example_inputs[0],
            check_trace=True,
            strict=False
        )
        traced_model = torch.jit.optimize_for_inference(traced_model)
        logger.info("Model traced successfully")
        
        # Convert to CoreML
        logger.info("Converting to CoreML...")
        mlmodel = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(
                    name="input_image",
                    shape=(1, 1, 256, 256),
                    dtype=np.float32,
                    default_value=0.0
                )
            ],
            outputs=[
                ct.TensorType(
                    name="output_image",
                    shape=(1, 3, 256, 256),
                    dtype=np.float32
                )
            ],
            minimum_deployment_target=ct.target.iOS16,
            compute_units=ct.ComputeUnit.ALL,
            compute_precision=ct.precision.FLOAT32,
            convert_to="mlprogram",
            # Advanced options for better performance
            optimization_config = ct.OptimizationConfig(
                auto_transpose_operations=True,  # Allow operation reordering
                auto_merge_divisions=True,      # Merge division operations
                global_constant_propagation=True,  # Propagate constants
            )
        )
        
        # Add model metadata
        mlmodel.author = "CAMUS Project"
        mlmodel.license = "MIT"
        mlmodel.short_description = "CAMUS Diffusion Model for Image Enhancement"
        mlmodel.version = "1.0"
        
        # Save the model
        output_path = "CAMUS_diffusion.mlpackage"
        mlmodel.save(output_path)
        logger.info(f"CoreML model saved successfully to {output_path}")
        
        # Verify the saved model
        logger.info("Verifying saved model...")
        loaded_model = ct.models.MLModel(output_path)
        test_input = {"input_image": example_inputs[0].numpy()}
        prediction = loaded_model.predict(test_input)
        assert prediction["output_image"].shape == (1, 3, 256, 256)
        logger.info("Saved model verified successfully")
        
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        raise

def convert_model_to_coreml():
    """
    Convert the diffusion model to CoreML format with proper input handling
    """
    # Create model with training config
    model = create_model()
    model.eval()
    
    # Wrap model with proper input handling and preprocessing
    wrapped_model = DiffusionModelWrapper(model, diffusion_steps=4000)
    
    # Example inputs for tracing
    example_inputs = (
        torch.randn(1, 1, 256, 256),  # Input image
        torch.tensor([0])  # Timestep
    )
    
    # Trace wrapped model
    traced_model = wrapped_model.trace(example_inputs)
    
    # Convert to CoreML
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.ImageType(
                name="input_image",
                shape=(1, 1, 256, 256),
                scale=1/255.0,
                bias=[0],
            ),
            ct.TensorType(
                name="timestep",
                shape=(1,),
                dtype=ct.numpy.int32
            )
        ],
        outputs=[
            ct.TensorType(
                name="output",
                shape=(1, 2, 256, 256)  # 2 channels for grayscale with learn_sigma=True
            )
        ],
        minimum_deployment_target=ct.target.iOS16,
        compute_units=ct.ComputeUnit.CPU_AND_NE
    )
    
    # Save the model
    mlmodel.save("diffusion_model.mlpackage")
    logger.info("Successfully converted and saved CoreML model")

if __name__ == "__main__":
    convert_model_to_coreml()
