#!/usr/bin/env python3
"""
CAMUS Diffusion Model to ONNX Conversion
========================================

This script converts the CAMUS diffusion model from PyTorch to ONNX format for iOS deployment.
Based on the successful segmentation model conversion methodology.

Model Analysis:
- Model Type: U-Net Diffusion Model with Self-Attention (RePaint-style)
- Parameters: 41.1M parameters
- Input: 3 channels (RGB), 256x256 resolution
- Output: 6 channels (learned sigma variant)
- Architecture: 24 input blocks + middle block + 24 output blocks
- Attention: 30 self-attention layers at resolutions 16x16 and 8x8
- Framework: Based on guided-diffusion (OpenAI)

Key Challenges for ONNX Conversion:
1. Time embeddings and conditioning
2. Self-attention layers with dynamic shapes
3. Skip connections across the U-Net
4. Complex diffusion sampling process
5. Large model size (41M parameters)
"""

import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
import sys
import os
from typing import Tuple, Dict, Any

# Add echogains to path
sys.path.insert(0, '/workspaces/Scrollshot_Fixer/echogains')

from echogains.RePaint.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    select_args,
)
from echogains.CONST import DEFAULT_CONFIG
import echogains.RePaint.conf_mgt as conf_mgt


class DiffusionONNXWrapper(torch.nn.Module):
    """
    Wrapper for the diffusion model to handle ONNX conversion.
    
    This wrapper simplifies the diffusion model interface for ONNX export by:
    1. Handling time embedding internally
    2. Providing a single forward pass
    3. Managing attention masks and conditioning
    """
    
    def __init__(self, diffusion_model, fixed_timestep: int = 125):
        super().__init__()
        self.model = diffusion_model
        self.fixed_timestep = fixed_timestep
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for ONNX export.
        
        Args:
            x: Input tensor of shape [batch_size, 3, 256, 256]
            
        Returns:
            Output tensor of shape [batch_size, 6, 256, 256]
        """
        batch_size = x.shape[0]
        
        # Create fixed timestep tensor
        t = torch.full((batch_size,), self.fixed_timestep, dtype=torch.long, device=x.device)
        
        # Forward pass through the model
        output = self.model(x, t)
        
        return output


def load_diffusion_model(model_path: str, device: str = 'cpu') -> torch.nn.Module:
    """
    Load the CAMUS diffusion model with the correct configuration.
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model on
        
    Returns:
        Loaded PyTorch model
    """
    print("Loading diffusion model...")
    
    # Use the default configuration from echogains
    config = DEFAULT_CONFIG.copy()
    config['model_path'] = model_path
    config['use_fp16'] = False  # Disable FP16 for ONNX export
    
    # Create proper configuration object
    conf_arg = conf_mgt.conf_base.Default_Conf()
    conf_arg.update(config)
    
    # Create model and diffusion
    model, diffusion = create_model_and_diffusion(
        **select_args(conf_arg, model_and_diffusion_defaults().keys()),
        conf=conf_arg
    )
    
    # Load the state dict
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Ensure model is in FP32 for ONNX export
    model.float()
    
    print(f"✓ Model loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def convert_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    input_shape: Tuple[int, int, int, int] = (1, 3, 256, 256),
    opset_version: int = 17
) -> bool:
    """
    Convert the diffusion model to ONNX format.
    
    Args:
        model: PyTorch model to convert
        output_path: Path to save the ONNX model
        input_shape: Input tensor shape (batch, channels, height, width)
        opset_version: ONNX opset version
        
    Returns:
        True if conversion successful, False otherwise
    """
    print(f"Converting model to ONNX...")
    print(f"  Input shape: {input_shape}")
    print(f"  Output path: {output_path}")
    
    try:
        # Create dummy input
        dummy_input = torch.randn(*input_shape)
        
        # Wrap the model for ONNX export
        wrapped_model = DiffusionONNXWrapper(model, fixed_timestep=125)
        wrapped_model.eval()
        
        # Test the wrapper
        print("Testing wrapped model...")
        with torch.no_grad():
            test_output = wrapped_model(dummy_input)
            print(f"  Wrapped model output shape: {test_output.shape}")
        
        # Export to ONNX
        print("Exporting to ONNX...")
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            verbose=False
        )
        
        print(f"✓ ONNX export successful")
        return True
        
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        return False


def validate_onnx_model(
    onnx_path: str,
    pytorch_model: torch.nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 3, 256, 256),
    tolerance: float = 1e-5
) -> bool:
    """
    Validate the ONNX model against the original PyTorch model.
    
    Args:
        onnx_path: Path to the ONNX model
        pytorch_model: Original PyTorch model
        input_shape: Input tensor shape for testing
        tolerance: Numerical tolerance for comparison
        
    Returns:
        True if validation passes, False otherwise
    """
    print("Validating ONNX model...")
    
    try:
        # Load ONNX model
        ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        # Create test input
        test_input = torch.randn(*input_shape)
        
        # Get PyTorch output
        wrapped_model = DiffusionONNXWrapper(pytorch_model, fixed_timestep=125)
        wrapped_model.eval()
        
        with torch.no_grad():
            pytorch_output = wrapped_model(test_input).numpy()
        
        # Get ONNX output
        onnx_output = ort_session.run(
            None,
            {'input': test_input.numpy()}
        )[0]
        
        # Compare outputs
        diff = np.abs(pytorch_output - onnx_output)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"  Output shape: {onnx_output.shape}")
        print(f"  Max difference: {max_diff:.8f}")
        print(f"  Mean difference: {mean_diff:.8f}")
        
        if max_diff < tolerance:
            print(f"✓ Validation passed (diff < {tolerance})")
            return True
        else:
            print(f"✗ Validation failed (diff >= {tolerance})")
            return False
            
    except Exception as e:
        print(f"✗ Validation error: {e}")
        return False


def get_model_info(onnx_path: str) -> Dict[str, Any]:
    """
    Get information about the ONNX model.
    
    Args:
        onnx_path: Path to the ONNX model
        
    Returns:
        Dictionary with model information
    """
    try:
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Get model size
        model_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        
        # Get input/output info
        input_info = []
        for input_tensor in onnx_model.graph.input:
            shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' 
                    for dim in input_tensor.type.tensor_type.shape.dim]
            input_info.append({
                'name': input_tensor.name,
                'shape': shape,
                'type': input_tensor.type.tensor_type.elem_type
            })
        
        output_info = []
        for output_tensor in onnx_model.graph.output:
            shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' 
                    for dim in output_tensor.type.tensor_type.shape.dim]
            output_info.append({
                'name': output_tensor.name,
                'shape': shape,
                'type': output_tensor.type.tensor_type.elem_type
            })
        
        return {
            'model_size_mb': model_size_mb,
            'inputs': input_info,
            'outputs': output_info,
            'opset_version': onnx_model.opset_import[0].version if onnx_model.opset_import else None
        }
        
    except Exception as e:
        print(f"Error getting model info: {e}")
        return {}


def main():
    """Main conversion function."""
    print("=" * 60)
    print("CAMUS DIFFUSION MODEL TO ONNX CONVERSION")
    print("=" * 60)
    print()
    
    # Configuration
    model_path = "/workspaces/Scrollshot_Fixer/CAMUS_diffusion_model.pt"
    onnx_output_path = "/workspaces/Scrollshot_Fixer/camus_diffusion_model.onnx"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"✗ Model file not found: {model_path}")
        return False
    
    print(f"Model path: {model_path}")
    print(f"Output path: {onnx_output_path}")
    print()
    
    try:
        # Step 1: Load the model
        model = load_diffusion_model(model_path, device='cpu')
        print()
        
        # Step 2: Convert to ONNX
        success = convert_to_onnx(model, onnx_output_path)
        if not success:
            return False
        print()
        
        # Step 3: Validate the conversion
        validation_success = validate_onnx_model(onnx_output_path, model)
        print()
        
        # Step 4: Get model information
        print("ONNX Model Information:")
        model_info = get_model_info(onnx_output_path)
        if model_info:
            print(f"  Size: {model_info['model_size_mb']:.1f} MB")
            print(f"  Opset version: {model_info.get('opset_version', 'Unknown')}")
            print(f"  Inputs: {model_info['inputs']}")
            print(f"  Outputs: {model_info['outputs']}")
        
        print()
        if validation_success:
            print("🎉 DIFFUSION MODEL CONVERSION COMPLETED SUCCESSFULLY!")
            print("The ONNX model is ready for iOS deployment.")
        else:
            print("⚠️  Conversion completed but validation failed.")
            print("Please review the model before deployment.")
        
        return validation_success
        
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
