#!/usr/bin/env python3
"""
Simple ONNX to CoreML Conversion
Author: GitHub Copilot
Purpose: Convert ONNX segmentation model to CoreML with workarounds for compatibility issues
"""

import os
import numpy as np
import onnxruntime as ort

def convert_with_workaround():
    """
    Convert ONNX to CoreML with environment workarounds
    """
    print("🚀 ONNX to CoreML Conversion (Workaround)")
    print("=" * 50)
    
    # Set environment variables for compatibility
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    
    onnx_path = "/workspaces/Scrollshot_Fixer/camus_segmentation_real_weights.onnx"
    coreml_path = "/workspaces/Scrollshot_Fixer/camus_segmentation.mlmodel"
    
    # Verify ONNX model
    print("1️⃣ Verifying ONNX model...")
    try:
        ort_session = ort.InferenceSession(onnx_path)
        input_info = ort_session.get_inputs()[0]
        output_info = ort_session.get_outputs()[0]
        
        print(f"✅ ONNX model verified")
        print(f"Input: {input_info.name} {input_info.shape}")
        print(f"Output: {output_info.name} {output_info.shape}")
        
        # Test inference
        dummy_input = np.random.randn(1, 1, 256, 256).astype(np.float32)
        output = ort_session.run(None, {input_info.name: dummy_input})[0]
        print(f"✅ ONNX inference: {output.shape}")
        
    except Exception as e:
        print(f"❌ ONNX verification failed: {e}")
        return False
    
    # Try different conversion approaches
    success = False
    
    # Approach 1: Direct onnx-coreml
    print("\n2️⃣ Trying onnx-coreml conversion...")
    try:
        from onnx_coreml import convert
        
        # Simple conversion
        mlmodel = convert(model=onnx_path)
        mlmodel.save(coreml_path)
        print("✅ onnx-coreml conversion successful!")
        success = True
        
    except Exception as e:
        print(f"❌ onnx-coreml failed: {e}")
    
    # Approach 2: Try with specific parameters
    if not success:
        print("\n3️⃣ Trying with specific parameters...")
        try:
            from onnx_coreml import convert
            
            # Convert with specific parameters
            mlmodel = convert(
                model=onnx_path,
                minimum_ios_deployment_target='13.0'
            )
            mlmodel.save(coreml_path)
            print("✅ Parameterized onnx-coreml conversion successful!")
            success = True
            
        except Exception as e:
            print(f"❌ Parameterized conversion failed: {e}")
    
    # Approach 3: PyTorch direct conversion
    if not success:
        print("\n4️⃣ Trying PyTorch to CoreML direct conversion...")
        try:
            import torch
            from load_actual_weights import main as load_weights
            
            # Load PyTorch model
            model = load_weights()
            if model is None:
                raise ValueError("Failed to load PyTorch model")
                
            model.eval()
            
            # Create traced model
            example_input = torch.randn(1, 1, 256, 256)
            traced_model = torch.jit.trace(model, example_input)
            
            # Try coremltools conversion
            import coremltools as ct
            
            mlmodel = ct.convert(
                traced_model,
                inputs=[ct.TensorType(shape=(1, 1, 256, 256), name="cardiac_image")]
            )
            
            mlmodel.save(coreml_path)
            print("✅ PyTorch to CoreML conversion successful!")
            success = True
            
        except Exception as e:
            print(f"❌ PyTorch direct conversion failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Verify the converted model
    if success:
        print("\n5️⃣ Verifying CoreML model...")
        try:
            import coremltools as ct
            
            # Load the model
            mlmodel = ct.models.MLModel(coreml_path)
            print("✅ CoreML model loaded successfully")
            
            # Get model info
            spec = mlmodel.get_spec()
            print(f"Model type: {spec.WhichOneof('Type')}")
            
            # Test prediction
            test_input = np.random.randn(1, 1, 256, 256).astype(np.float32)
            
            # Prepare input (CoreML expects different format)
            input_name = list(mlmodel.get_spec().description.input)[0].name
            input_dict = {input_name: test_input}
            
            # Run prediction
            coreml_output = mlmodel.predict(input_dict)
            output_key = list(coreml_output.keys())[0]
            output_array = coreml_output[output_key]
            
            print(f"✅ CoreML inference successful!")
            print(f"Output shape: {output_array.shape}")
            print(f"Output range: [{np.min(output_array):.4f}, {np.max(output_array):.4f}]")
            
            # Compare with ONNX
            onnx_output = ort_session.run(None, {input_info.name: test_input})[0]
            max_diff = np.max(np.abs(output_array - onnx_output))
            print(f"Max difference ONNX vs CoreML: {max_diff:.6f}")
            
            print(f"\n🎉 SUCCESS! CoreML model saved: {coreml_path}")
            
        except Exception as e:
            print(f"❌ CoreML verification failed: {e}")
            import traceback
            traceback.print_exc()
            success = False
    
    return success

def main():
    """Main function"""
    print("🔧 Setting up environment for CoreML conversion...")
    
    success = convert_with_workaround()
    
    if success:
        print("\n✅ Segmentation model successfully converted to CoreML!")
        print("📁 Files available:")
        print("   - ONNX: camus_segmentation_real_weights.onnx")
        print("   - CoreML: camus_segmentation.mlmodel")
        print("\n🎯 Ready for iOS deployment!")
    else:
        print("\n❌ CoreML conversion failed")
        print("💡 The ONNX model is still available for other platforms")
    
    return success

if __name__ == "__main__":
    main()
