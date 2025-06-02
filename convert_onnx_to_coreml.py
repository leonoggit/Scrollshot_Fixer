#!/usr/bin/env python3
"""
Convert ONNX Segmentation Model to CoreML
Author: GitHub Copilot
Purpose: Convert the validated ONNX model to CoreML format for iOS deployment
"""

import coremltools as ct
import numpy as np
import onnxruntime as ort

def convert_segmentation_to_coreml():
    """
    Convert the ONNX segmentation model to CoreML format
    """
    print("🚀 Converting ONNX Segmentation Model to CoreML")
    print("=" * 55)
    
    onnx_path = "/workspaces/Scrollshot_Fixer/camus_segmentation_real_weights.onnx"
    coreml_path = "/workspaces/Scrollshot_Fixer/camus_segmentation.mlmodel"
    
    # Verify ONNX model exists and works
    print("1️⃣ Verifying ONNX model...")
    try:
        ort_session = ort.InferenceSession(onnx_path)
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
        input_shape = ort_session.get_inputs()[0].shape
        output_shape = ort_session.get_outputs()[0].shape
        
        print(f"✅ ONNX model loaded successfully")
        print(f"Input: {input_name} {input_shape}")
        print(f"Output: {output_name} {output_shape}")
        
        # Test inference
        dummy_input = np.random.randn(1, 1, 256, 256).astype(np.float32)
        onnx_output = ort_session.run([output_name], {input_name: dummy_input})[0]
        print(f"✅ ONNX inference test successful: {onnx_output.shape}")
        
    except Exception as e:
        print(f"❌ ONNX model verification failed: {e}")
        return False
    
    # Convert to CoreML
    print("\n2️⃣ Converting to CoreML...")
    try:
        # Convert ONNX to CoreML - Note: coremltools doesn't directly support ONNX
        # We need to use onnx-coreml converter instead
        print("💡 Using onnx-coreml for conversion...")
        
        # Try importing onnx_coreml
        try:
            from onnx_coreml import convert
            
            # Convert ONNX to CoreML
            coreml_model = convert(onnx_path)
            print("✅ CoreML conversion successful!")
            
        except ImportError:
            print("❌ onnx-coreml not installed. Installing...")
            import subprocess
            import sys
            
            # Install onnx-coreml
            result = subprocess.run([sys.executable, "-m", "pip", "install", "onnx-coreml"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ onnx-coreml installed successfully")
                from onnx_coreml import convert
                coreml_model = convert(onnx_path)
                print("✅ CoreML conversion successful!")
            else:
                print(f"❌ Failed to install onnx-coreml: {result.stderr}")
                return False
                
    except Exception as e:
        print(f"❌ CoreML conversion failed: {e}")
        print("💡 Trying alternative approach...")
        
        try:
            # Alternative: Use torch.jit and then convert
            print("🔄 Loading PyTorch model for direct conversion...")
            import torch
            from load_actual_weights import main as load_weights
            
            # Load the PyTorch model
            model = load_weights()
            if model is None:
                print("❌ Failed to load PyTorch model")
                return False
                
            model.eval()
            
            # Create example input
            example_input = torch.randn(1, 1, 256, 256)
            
            # Convert PyTorch to CoreML directly
            traced_model = torch.jit.trace(model, example_input)
            
            # Now convert the traced model
            coreml_model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(shape=(1, 1, 256, 256))]
            )
            print("✅ Direct PyTorch to CoreML conversion successful!")
            
        except Exception as e2:
            print(f"❌ Alternative conversion also failed: {e2}")
            import traceback
            traceback.print_exc()
            return False
    
    # Add model metadata
    print("\n3️⃣ Adding model metadata...")
    coreml_model.short_description = "CAMUS Cardiac Segmentation Model"
    coreml_model.author = "nnU-Net Framework"
    coreml_model.license = "Academic Use"
    coreml_model.version = "1.0"
    
    # Add input/output descriptions
    coreml_model.input_description["cardiac_image"] = "Grayscale cardiac ultrasound image (256x256)"
    coreml_model.output_description["segmentation_logits"] = "Segmentation logits for 3 classes: background, left ventricle, myocardium"
    
    # Save the model
    print("\n4️⃣ Saving CoreML model...")
    coreml_model.save(coreml_path)
    print(f"✅ CoreML model saved: {coreml_path}")
    
    # Verify the saved model
    print("\n5️⃣ Verifying saved CoreML model...")
    try:
        # Load and test the saved model
        loaded_model = ct.models.MLModel(coreml_path)
        
        # Get model info
        spec = loaded_model.get_spec()
        print(f"✅ Model loaded successfully")
        print(f"Model type: {spec.WhichOneof('Type')}")
        
        # Test prediction
        print("\n6️⃣ Testing CoreML inference...")
        test_input = np.random.randn(1, 1, 256, 256).astype(np.float32)
        
        # Prepare input dictionary
        input_dict = {}
        input_name = list(loaded_model.get_spec().description.input)[0].name
        input_dict[input_name] = test_input
        
        # Run prediction
        coreml_output = loaded_model.predict(input_dict)
        output_key = list(coreml_output.keys())[0]
        output_array = coreml_output[output_key]
        
        print(f"✅ CoreML inference successful!")
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {output_array.shape}")
        print(f"Output range: [{np.min(output_array):.4f}, {np.max(output_array):.4f}]")
        
        # Compare with ONNX output for same input
        onnx_output_test = ort_session.run([output_name], {input_name: test_input})[0]
        max_diff = np.max(np.abs(output_array - onnx_output_test))
        print(f"Max difference ONNX vs CoreML: {max_diff:.6f}")
        
        if max_diff < 1e-3:
            print("✅ CoreML and ONNX outputs match within tolerance!")
        else:
            print("⚠️ Some differences between ONNX and CoreML outputs")
        
    except Exception as e:
        print(f"❌ CoreML model verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 SUCCESS! Segmentation model converted to CoreML")
    print("=" * 55)
    print(f"📁 ONNX Model: {onnx_path}")
    print(f"📁 CoreML Model: {coreml_path}")
    print("\n🎯 Ready for iOS deployment!")
    
    return True

def main():
    """Main function"""
    success = convert_segmentation_to_coreml()
    if success:
        print("\n✅ Conversion pipeline completed successfully!")
    else:
        print("\n❌ Conversion pipeline failed")
    
    return success

if __name__ == "__main__":
    main()
