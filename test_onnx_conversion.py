#!/usr/bin/env python3
"""
Convert Reconstructed nnU-Net to ONNX
Author: GitHub Copilot
Purpose: Test ONNX conversion with our reconstructed model before dealing with weight loading
"""

import torch
import torch.onnx
import numpy as np

# Import our reconstructed model
from reconstruct_nnunet import ReconstructednnUNet, test_reconstructed_model

def convert_to_onnx(model, output_path="nnunet_segmentation.onnx"):
    """
    Convert the model to ONNX format
    """
    print(f"🔄 Converting model to ONNX: {output_path}")
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 1, 256, 256)
    
    try:
        # Export to ONNX
        torch.onnx.export(
            model,                      # model being run
            dummy_input,                # model input (or a tuple for multiple inputs)
            output_path,                # where to save the model
            export_params=True,         # store the trained parameter weights inside the model file
            opset_version=11,           # the ONNX version to export the model to
            do_constant_folding=True,   # whether to execute constant folding for optimization
            input_names=['input'],      # the model's input names
            output_names=['output'],    # the model's output names
            dynamic_axes={
                'input': {0: 'batch_size'},    # variable length axes
                'output': {0: 'batch_size'}
            }
        )
        
        print("✅ ONNX export successful!")
        return True
        
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_onnx_model(onnx_path="nnunet_segmentation.onnx"):
    """
    Test the exported ONNX model
    """
    print(f"🧪 Testing ONNX model: {onnx_path}")
    
    try:
        import onnxruntime as ort
        
        # Create inference session
        ort_session = ort.InferenceSession(onnx_path)
        
        # Get input/output info
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
        
        print(f"Input name: {input_name}")
        print(f"Output name: {output_name}")
        
        # Test inference
        test_input = np.random.randn(1, 1, 256, 256).astype(np.float32)
        result = ort_session.run([output_name], {input_name: test_input})
        
        print(f"✅ ONNX inference successful!")
        print(f"Output shape: {result[0].shape}")
        print(f"Output range: [{result[0].min():.3f}, {result[0].max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ ONNX inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main conversion function
    """
    print("🚀 nnU-Net to ONNX Conversion Test")
    print("=" * 40)
    
    # Create and test the reconstructed model
    print("1️⃣ Creating reconstructed model...")
    success, model = test_reconstructed_model()
    
    if not success:
        print("❌ Model creation failed. Cannot proceed with conversion.")
        return
    
    # Convert to ONNX
    print("\n2️⃣ Converting to ONNX...")
    onnx_success = convert_to_onnx(model)
    
    if not onnx_success:
        print("❌ ONNX conversion failed.")
        return
    
    # Test ONNX model
    print("\n3️⃣ Testing ONNX model...")
    test_success = test_onnx_model()
    
    if test_success:
        print("\n🎉 SUCCESS! Full pipeline working:")
        print("✅ Model reconstruction")
        print("✅ ONNX conversion")
        print("✅ ONNX inference")
        print("\n🎯 Next: Load actual weights and convert with real parameters")
    else:
        print("\n❌ ONNX testing failed.")

if __name__ == "__main__":
    main()
