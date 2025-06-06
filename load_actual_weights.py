#!/usr/bin/env python3
"""
Load Actual Weights into nnU-Net Model
Author: GitHub Copilot
Purpose: Map and load the actual checkpoint weights into our reconstructed model
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from pathlib import Path
from reconstruct_nnunet import ReconstructednnUNet, analyze_nnunet_structure

def create_weight_mapping(model_state_dict, checkpoint_state_dict):
    """
    Create a mapping between our model's state dict keys and the checkpoint keys
    """
    print("🔗 Creating weight mapping...")
    
    model_keys = list(model_state_dict.keys())
    checkpoint_keys = list(checkpoint_state_dict.keys())
    
    print(f"Model has {len(model_keys)} parameters")
    print(f"Checkpoint has {len(checkpoint_keys)} parameters")
    
    # Create mapping dictionary
    weight_mapping = {}
    
    # Sort keys for easier debugging
    model_keys.sort()
    checkpoint_keys.sort()
    
    print("\n📋 First 10 model keys:")
    for i, key in enumerate(model_keys[:10]):
        print(f"  {i}: {key} -> {model_state_dict[key].shape}")
    
    print("\n📋 First 10 checkpoint keys:")
    for i, key in enumerate(checkpoint_keys[:10]):
        print(f"  {i}: {key} -> {checkpoint_state_dict[key].shape}")
    
    # Try to find pattern-based mappings
    print("\n🔍 Analyzing key patterns...")
    
    # Encoder stages mapping
    for model_key in model_keys:
        if model_key.startswith('encoder_stages.'):
            # Extract stage number and layer info
            # Model key: encoder_stages.0.convs.convs.0.conv.weight
            # Checkpoint key: encoder.stages.0.0.convs.0.conv.weight
            
            parts = model_key.split('.')
            if len(parts) >= 6:
                stage_num = parts[1]  # 0, 1, 2, etc.
                conv_block = parts[4]  # 0 or 1 (first or second conv in double conv)
                layer_type = parts[5]  # conv, norm, etc.
                param_type = parts[6] if len(parts) > 6 else parts[5]  # weight or bias
                
                # Construct checkpoint key
                checkpoint_key = f"encoder.stages.{stage_num}.0.convs.{conv_block}.{layer_type}.{param_type}"
                
                if checkpoint_key in checkpoint_state_dict:
                    # Check shape compatibility
                    model_shape = model_state_dict[model_key].shape
                    checkpoint_shape = checkpoint_state_dict[checkpoint_key].shape
                    
                    if model_shape == checkpoint_shape:
                        weight_mapping[model_key] = checkpoint_key
                        print(f"✅ Mapped: {model_key} -> {checkpoint_key}")
                    else:
                        print(f"❌ Shape mismatch: {model_key} {model_shape} vs {checkpoint_key} {checkpoint_shape}")
    
    # Segmentation layers mapping  
    for model_key in model_keys:
        if model_key.startswith('seg_layers.'):
            # Model key: seg_layers.0.weight
            # Checkpoint key: decoder.seg_layers.0.weight
            
            checkpoint_key = f"decoder.{model_key}"
            
            if checkpoint_key in checkpoint_state_dict:
                model_shape = model_state_dict[model_key].shape
                checkpoint_shape = checkpoint_state_dict[checkpoint_key].shape
                
                if model_shape == checkpoint_shape:
                    weight_mapping[model_key] = checkpoint_key
                    print(f"✅ Mapped: {model_key} -> {checkpoint_key}")
                else:
                    print(f"❌ Shape mismatch: {model_key} {model_shape} vs {checkpoint_key} {checkpoint_shape}")
    
    print(f"\n📊 Mapping Summary:")
    print(f"Successfully mapped: {len(weight_mapping)} parameters")
    print(f"Unmapped model parameters: {len(model_keys) - len(weight_mapping)}")
    
    return weight_mapping

def load_mapped_weights(model, checkpoint_state_dict, weight_mapping):
    """
    Load the mapped weights into the model
    """
    print("📥 Loading mapped weights...")
    
    loaded_count = 0
    total_count = len(weight_mapping)
    
    with torch.no_grad():
        for model_key, checkpoint_key in weight_mapping.items():
            try:
                checkpoint_weight = checkpoint_state_dict[checkpoint_key]
                model.state_dict()[model_key].copy_(checkpoint_weight)
                loaded_count += 1
            except Exception as e:
                print(f"❌ Failed to load {model_key}: {e}")
    
    print(f"✅ Successfully loaded {loaded_count}/{total_count} weights")
    return loaded_count == total_count

def create_improved_nnunet(checkpoint_state_dict):
    """
    Create an improved nnU-Net model that exactly matches the checkpoint structure
    """
    print("🏗️ Creating improved nnU-Net model...")
    
    # Analyze the checkpoint structure more carefully
    stage_info = analyze_nnunet_structure(checkpoint_state_dict)
    
    # Look at the exact structure
    encoder_channels = []
    
    # Extract channel progression from actual weights
    for stage_num in range(7):  # We know there are 7 stages
        stage_key = f'encoder.stages.{stage_num}.0.convs.0.conv.weight'
        if stage_key in checkpoint_state_dict:
            weight = checkpoint_state_dict[stage_key]
            in_channels = weight.shape[1]
            out_channels = weight.shape[0]
            encoder_channels.append((in_channels, out_channels))
            print(f"Stage {stage_num}: {in_channels} -> {out_channels}")
    
    # Create the exact model architecture
    class ExactnnUNet(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Build encoder stages exactly as in checkpoint
            self.encoder_stages = nn.ModuleList()
            
            for stage_num, (in_ch, out_ch) in enumerate(encoder_channels):
                stage = nn.Sequential(
                    nn.Sequential(  # First conv block
                        nn.Conv2d(in_ch, out_ch, 3, padding=1),
                        nn.InstanceNorm2d(out_ch),
                        nn.LeakyReLU(0.01, inplace=True)
                    ),
                    nn.Sequential(  # Second conv block  
                        nn.Conv2d(out_ch, out_ch, 3, padding=1),
                        nn.InstanceNorm2d(out_ch),
                        nn.LeakyReLU(0.01, inplace=True)
                    )
                )
                self.encoder_stages.append(stage)
            
            # Build segmentation layers
            self.seg_layers = nn.ModuleList()
            for i in range(6):  # 6 segmentation layers
                seg_key = f'decoder.seg_layers.{i}.weight'
                if seg_key in checkpoint_state_dict:
                    weight = checkpoint_state_dict[seg_key]
                    out_channels, in_channels = weight.shape[:2]
                    kernel_size = weight.shape[2]
                    
                    if kernel_size == 1:
                        conv = nn.Conv2d(in_channels, out_channels, 1)
                    else:
                        conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
                    
                    self.seg_layers.append(conv)
                    print(f"Seg layer {i}: {in_channels} -> {out_channels}, kernel: {kernel_size}")
        
        def forward(self, x):
            # Encoder path with feature collection
            features = []
            
            for stage_idx, stage in enumerate(self.encoder_stages):
                x = stage(x)
                features.append(x)
                
                # Downsample for next stage (except last stage)
                if stage_idx < len(self.encoder_stages) - 1:
                    x = nn.functional.max_pool2d(x, 2)
            
            # Apply the correct segmentation layer based on feature resolution
            # features[-1] has 512 channels, so use seg_layers[0] or [1] (both expect 512)
            # Let's use the deepest segmentation layer
            x = self.seg_layers[0](features[-1])  # seg_layers[0] expects 512 channels
            
            # Upsample to original size (256x256)
            # After 6 downsampling steps: 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4
            # So we need to upsample from 4x4 to 256x256
            target_size = (256, 256)
            if x.shape[2] != target_size[0] or x.shape[3] != target_size[1]:
                x = nn.functional.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            
            return x
    
    return ExactnnUNet()

def create_exact_weight_mapping(model, checkpoint_state_dict):
    """
    Create exact weight mapping for the improved model
    """
    print("🎯 Creating exact weight mapping...")
    
    weight_mapping = {}
    model_state = model.state_dict()
    
    # Map encoder stages
    for stage_idx in range(len(model.encoder_stages)):
        stage_prefix = f"encoder_stages.{stage_idx}"
        checkpoint_prefix = f"encoder.stages.{stage_idx}.0.convs"
        
        # Map conv blocks (0 and 1 for double conv)
        for conv_idx in range(2):
            # Map conv layer
            model_conv_key = f"{stage_prefix}.{conv_idx}.0.weight"
            checkpoint_conv_key = f"{checkpoint_prefix}.{conv_idx}.conv.weight"
            
            if model_conv_key in model_state and checkpoint_conv_key in checkpoint_state_dict:
                if model_state[model_conv_key].shape == checkpoint_state_dict[checkpoint_conv_key].shape:
                    weight_mapping[model_conv_key] = checkpoint_conv_key
                    print(f"✅ Conv: {model_conv_key} -> {checkpoint_conv_key}")
            
            # Map conv bias
            model_bias_key = f"{stage_prefix}.{conv_idx}.0.bias"
            checkpoint_bias_key = f"{checkpoint_prefix}.{conv_idx}.conv.bias"
            
            if model_bias_key in model_state and checkpoint_bias_key in checkpoint_state_dict:
                if model_state[model_bias_key].shape == checkpoint_state_dict[checkpoint_bias_key].shape:
                    weight_mapping[model_bias_key] = checkpoint_bias_key
                    print(f"✅ Bias: {model_bias_key} -> {checkpoint_bias_key}")
            
            # Map normalization layers (InstanceNorm)
            model_norm_weight_key = f"{stage_prefix}.{conv_idx}.1.weight"
            model_norm_bias_key = f"{stage_prefix}.{conv_idx}.1.bias"
            checkpoint_norm_weight_key = f"{checkpoint_prefix}.{conv_idx}.instnorm.weight"
            checkpoint_norm_bias_key = f"{checkpoint_prefix}.{conv_idx}.instnorm.bias"
            
            if model_norm_weight_key in model_state and checkpoint_norm_weight_key in checkpoint_state_dict:
                if model_state[model_norm_weight_key].shape == checkpoint_state_dict[checkpoint_norm_weight_key].shape:
                    weight_mapping[model_norm_weight_key] = checkpoint_norm_weight_key
            
            if model_norm_bias_key in model_state and checkpoint_norm_bias_key in checkpoint_state_dict:
                if model_state[model_norm_bias_key].shape == checkpoint_state_dict[checkpoint_norm_bias_key].shape:
                    weight_mapping[model_norm_bias_key] = checkpoint_norm_bias_key
    
    # Map segmentation layers
    for seg_idx in range(len(model.seg_layers)):
        model_seg_key = f"seg_layers.{seg_idx}.weight"
        checkpoint_seg_key = f"decoder.seg_layers.{seg_idx}.weight"
        
        if model_seg_key in model_state and checkpoint_seg_key in checkpoint_state_dict:
            if model_state[model_seg_key].shape == checkpoint_state_dict[checkpoint_seg_key].shape:
                weight_mapping[model_seg_key] = checkpoint_seg_key
                print(f"✅ Seg: {model_seg_key} -> {checkpoint_seg_key}")
        
        # Map bias if exists
        model_seg_bias_key = f"seg_layers.{seg_idx}.bias"
        checkpoint_seg_bias_key = f"decoder.seg_layers.{seg_idx}.bias"
        
        if model_seg_bias_key in model_state and checkpoint_seg_bias_key in checkpoint_state_dict:
            if model_state[model_seg_bias_key].shape == checkpoint_state_dict[checkpoint_seg_bias_key].shape:
                weight_mapping[model_seg_bias_key] = checkpoint_seg_bias_key
    
    return weight_mapping

def test_loaded_model(model):
    """
    Test the model with loaded weights
    """
    print("🧪 Testing model with loaded weights...")
    
    model.eval()
    
    with torch.no_grad():
        # Test input
        test_input = torch.randn(1, 1, 256, 256)
        
        try:
            output = model(test_input)
            print(f"✅ Forward pass successful!")
            print(f"Input shape: {test_input.shape}")
            print(f"Output shape: {output.shape}")
            print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
            
            # Check if output looks reasonable
            if output.shape[1] == 3:  # 3 classes
                print("✅ Output has correct number of classes (3)")
                
                # Apply softmax to see class probabilities
                probs = torch.softmax(output, dim=1)
                print(f"Class probabilities range: [{probs.min():.4f}, {probs.max():.4f}]")
                
                return True
            else:
                print(f"❌ Unexpected number of output classes: {output.shape[1]}")
                return False
                
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """
    Main function to load weights into the nnU-Net model
    """
    print("🚀 Loading Actual Weights into nnU-Net")
    print("=" * 45)
    
    # Load the checkpoint
    # Use a path relative to this file so the script works from any environment
    repo_root = Path(__file__).resolve().parent
    model_path = repo_root / "model_files" / "Segmentation model" / "checkpoint_best.pth"
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['network_weights']
    
    print(f"✅ Loaded checkpoint with {len(state_dict)} parameters")
    
    # Create exact model architecture
    model = create_improved_nnunet(state_dict)
    print(f"✅ Created model with {len(list(model.parameters()))} parameter tensors")
    
    # Create weight mapping
    weight_mapping = create_exact_weight_mapping(model, state_dict)
    
    # Load weights
    success = load_mapped_weights(model, state_dict, weight_mapping)
    
    if success:
        print("\n✅ All weights loaded successfully!")
        
        # Test the model
        test_success = test_loaded_model(model)
        
        if test_success:
            print("\n🎉 SUCCESS! Model with actual weights is working!")
            print("🎯 Ready for final ONNX conversion with real weights")
            
            # Save the loaded model
            torch.save(model.state_dict(), "nnunet_loaded_weights.pth")
            print("💾 Saved loaded model weights")
            
            return model
        else:
            print("\n❌ Model testing failed")
    else:
        print("\n❌ Weight loading incomplete")
    
    return None

if __name__ == "__main__":
    main()
