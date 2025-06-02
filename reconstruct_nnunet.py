#!/usr/bin/env python3
"""
Reconstruct nnU-Net Segmentation Model Architecture
Author: GitHub Copilot
Purpose: Build the actual model architecture from the checkpoint state_dict
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

def analyze_nnunet_structure(state_dict):
    """
    Analyze the nnU-Net state dict to understand the architecture
    """
    print("🔍 Analyzing nnU-Net Structure...")
    
    # Find all unique layer patterns
    encoder_stages = set()
    decoder_stages = set()
    
    for key in state_dict.keys():
        if key.startswith('encoder.stages.'):
            # Extract stage number
            parts = key.split('.')
            if len(parts) >= 3:
                stage_num = parts[2]
                encoder_stages.add(int(stage_num))
        elif key.startswith('decoder.'):
            parts = key.split('.')
            if 'seg_layers' in key:
                # Segmentation layers
                if len(parts) >= 3:
                    layer_num = parts[2]
                    if layer_num.isdigit():
                        decoder_stages.add(f"seg_layer_{layer_num}")
            else:
                # Regular decoder stages
                if len(parts) >= 2:
                    decoder_stages.add(parts[1])
    
    print(f"Encoder stages: {sorted(encoder_stages)}")
    print(f"Decoder components: {sorted(decoder_stages)}")
    
    # Analyze conv layer dimensions for each stage
    stage_info = {}
    for stage in sorted(encoder_stages):
        stage_key_pattern = f'encoder.stages.{stage}.'
        stage_layers = [k for k in state_dict.keys() if k.startswith(stage_key_pattern)]
        
        # Find conv layers
        conv_layers = []
        for layer in stage_layers:
            if 'conv.weight' in layer:
                weight = state_dict[layer]
                conv_layers.append((layer, weight.shape))
        
        stage_info[f'encoder_stage_{stage}'] = conv_layers
    
    # Analyze decoder/segmentation layers
    seg_layers = []
    for key, value in state_dict.items():
        if 'decoder.seg_layers' in key and 'weight' in key:
            seg_layers.append((key, value.shape))
    
    stage_info['segmentation_layers'] = seg_layers
    
    return stage_info

class ConvBlock(nn.Module):
    """
    Basic convolutional block used in nnU-Net
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_channels)  # nnU-Net typically uses InstanceNorm
        self.activation = nn.LeakyReLU(0.01, inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class DoubleConv(nn.Module):
    """
    Double convolution block (common in U-Net architectures)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convs = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )
    
    def forward(self, x):
        return self.convs(x)

class EncoderStage(nn.Module):
    """
    Encoder stage with convolutions and optional downsampling
    """
    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()
        self.convs = DoubleConv(in_channels, out_channels)
        self.downsample = nn.MaxPool2d(2) if downsample else None
    
    def forward(self, x):
        features = self.convs(x)
        if self.downsample:
            downsampled = self.downsample(features)
            return features, downsampled
        return features

class DecoderStage(nn.Module):
    """
    Decoder stage with upsampling and convolutions
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.convs = DoubleConv(in_channels // 2 + skip_channels, out_channels)
    
    def forward(self, x, skip):
        x = self.upsample(x)
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        x = self.convs(x)
        return x

class ReconstructednnUNet(nn.Module):
    """
    Reconstructed nnU-Net model based on the state dict analysis
    """
    def __init__(self, in_channels=1, num_classes=3):
        super().__init__()
        
        # Based on the analysis: first conv is [32, 1, 3, 3]
        # This suggests the architecture progression
        
        # Encoder path
        self.encoder_stages = nn.ModuleList([
            EncoderStage(1, 32, downsample=True),    # Stage 0
            EncoderStage(32, 64, downsample=True),   # Stage 1  
            EncoderStage(64, 128, downsample=True),  # Stage 2
            EncoderStage(128, 256, downsample=True), # Stage 3
            EncoderStage(256, 512, downsample=False) # Stage 4 (bottleneck)
        ])
        
        # Decoder path
        self.decoder_stages = nn.ModuleList([
            DecoderStage(512, 256, 256),  # Up from bottleneck
            DecoderStage(256, 128, 128),  # Stage 3 up
            DecoderStage(128, 64, 64),    # Stage 2 up
            DecoderStage(64, 32, 32),     # Stage 1 up
        ])
        
        # Segmentation head - based on analysis: final is [3, 32, 1, 1]
        self.seg_layers = nn.ModuleList([
            nn.Conv2d(32, 32, 3, padding=1),   # Layer 0
            nn.Conv2d(32, 32, 3, padding=1),   # Layer 1
            nn.Conv2d(32, 32, 3, padding=1),   # Layer 2
            nn.Conv2d(32, 32, 3, padding=1),   # Layer 3
            nn.Conv2d(32, 32, 3, padding=1),   # Layer 4
            nn.Conv2d(32, num_classes, 1),     # Layer 5 - final output
        ])
    
    def forward(self, x):
        # Store skip connections
        skip_connections = []
        
        # Encoder path
        for i, stage in enumerate(self.encoder_stages[:-1]):
            skip, x = stage(x)
            skip_connections.append(skip)
        
        # Bottleneck
        x = self.encoder_stages[-1](x)
        
        # Decoder path
        skip_connections.reverse()
        for i, (decoder_stage, skip) in enumerate(zip(self.decoder_stages, skip_connections)):
            x = decoder_stage(x, skip)
        
        # Segmentation head
        for i, seg_layer in enumerate(self.seg_layers[:-1]):
            x = F.relu(seg_layer(x))
        
        # Final output (no activation - logits)
        x = self.seg_layers[-1](x)
        
        return x

def load_weights_into_model(model, state_dict):
    """
    Load the actual weights from the checkpoint into our reconstructed model
    """
    print("📥 Loading weights into reconstructed model...")
    
    try:
        # This is tricky - we need to map the checkpoint keys to our model keys
        # For now, let's see what keys we have in both
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(state_dict.keys())
        
        print(f"Model has {len(model_keys)} parameters")
        print(f"Checkpoint has {len(checkpoint_keys)} parameters")
        
        # Try to find matching keys
        matching_keys = model_keys.intersection(checkpoint_keys)
        print(f"Directly matching keys: {len(matching_keys)}")
        
        if len(matching_keys) > 0:
            print("Some matching keys found:")
            for key in list(matching_keys)[:5]:  # Show first 5
                print(f"  {key}: {model.state_dict()[key].shape} vs {state_dict[key].shape}")
        
        # For now, return the model without loading weights
        # We'll need to do careful key mapping
        print("⚠️  Weight loading requires careful key mapping - skipping for now")
        return model
        
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return model

def test_reconstructed_model():
    """
    Test the reconstructed model with dummy input
    """
    print("\n🧪 Testing Reconstructed Model...")
    
    # Create model
    model = ReconstructednnUNet(in_channels=1, num_classes=3)
    model.eval()
    
    # Test input
    test_input = torch.randn(1, 1, 256, 256)
    print(f"Test input shape: {test_input.shape}")
    
    try:
        with torch.no_grad():
            output = model(test_input)
            print(f"✅ Model output shape: {output.shape}")
            print(f"Expected: (1, 3, 256, 256)")
            
            # Test if it's the right shape
            if output.shape == (1, 3, 256, 256):
                print("✅ Output shape matches expected segmentation output!")
                return True, model
            else:
                print("❌ Output shape doesn't match expected")
                return False, model
                
    except Exception as e:
        print(f"❌ Model forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False, model

def main():
    """
    Main function to reconstruct and test the nnU-Net model
    """
    print("🚀 nnU-Net Model Reconstruction")
    print("=" * 40)
    
    # Load the checkpoint
    model_path = "/workspaces/Scrollshot_Fixer/model_files/Segmentation model/checkpoint_best.pth"
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['network_weights']
    
    # Analyze structure
    stage_info = analyze_nnunet_structure(state_dict)
    
    print("\n📊 Stage Information:")
    for stage, layers in stage_info.items():
        print(f"\n{stage}:")
        for layer_name, shape in layers:
            print(f"  {layer_name}: {shape}")
    
    # Test our reconstructed model
    success, model = test_reconstructed_model()
    
    if success:
        print("\n✅ Model reconstruction successful!")
        print("🎯 Next step: Proper weight loading and conversion to CoreML/ONNX")
    else:
        print("\n❌ Model reconstruction needs refinement")
    
    return model, state_dict, stage_info

if __name__ == "__main__":
    main()
