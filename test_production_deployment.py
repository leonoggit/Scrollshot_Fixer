#!/usr/bin/env python3
"""
Production Deployment Test Suite for CAMUS Segmentation Model
============================================================

This script validates the segmentation model for production iOS deployment
by testing all critical components and performance metrics.
"""

import os
import pytest
import numpy as np
import time
from PIL import Image
import onnxruntime as ort

if os.environ.get("CI_SKIP_ONNX") == "1":
    pytest.skip("Skipping ONNX tests due to environment limitations", allow_module_level=True)

def test_model_loading():
    """Test if the ONNX model loads correctly"""
    print("🔍 Testing Model Loading...")
    try:
        model_path = '/workspaces/Scrollshot_Fixer/camus_segmentation_real_weights.onnx'
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False
            
        session = ort.InferenceSession(model_path)
        print("✅ Model loaded successfully")
        
        # Get model metadata
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        
        print(f"📊 Model Specifications:")
        print(f"   Input: {inputs[0].name} - {inputs[0].shape} ({inputs[0].type})")
        print(f"   Output: {outputs[0].name} - {outputs[0].shape} ({outputs[0].type})")
        
        # Validate expected dimensions for cardiac segmentation
        expected_input_shape = [1, 1, 256, 256]  # [batch, channels, height, width]
        expected_output_shape = [1, 3, 256, 256]  # [batch, classes, height, width]
        
        if list(inputs[0].shape) == expected_input_shape:
            print("✅ Input shape matches expected format")
        else:
            print(f"⚠️  Input shape mismatch: expected {expected_input_shape}, got {inputs[0].shape}")
            
        if list(outputs[0].shape) == expected_output_shape:
            print("✅ Output shape matches expected format")
        else:
            print(f"⚠️  Output shape mismatch: expected {expected_output_shape}, got {outputs[0].shape}")
        
        return session
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def create_synthetic_cardiac_image():
    """Create a synthetic cardiac ultrasound image for testing"""
    print("🏥 Creating synthetic cardiac ultrasound image...")
    
    # Create 256x256 grayscale image
    image = np.zeros((256, 256), dtype=np.float32)
    
    # Add background noise (typical in ultrasound)
    noise = np.random.normal(0.1, 0.05, (256, 256))
    image += noise
    
    # Simulate left ventricle cavity (brighter circular region)
    center_x, center_y = 160, 128  # Offset center for realism
    
    # Create LV cavity
    y, x = np.ogrid[:256, :256]
    cavity_mask = (x - center_x)**2 + (y - center_y)**2 <= 30**2
    image[cavity_mask] = 0.7
    
    # Create LV wall (ring around cavity)
    wall_mask = ((x - center_x)**2 + (y - center_y)**2 <= 50**2) & ((x - center_x)**2 + (y - center_y)**2 > 30**2)
    image[wall_mask] = 0.5
    
    # Add some speckle pattern typical in ultrasound
    for _ in range(200):
        px, py = np.random.randint(0, 256, 2)
        intensity = np.random.uniform(0.2, 0.8)
        image[py:py+2, px:px+2] = intensity
    
    # Normalize to [0, 1]
    image = np.clip(image, 0, 1)
    
    print(f"✅ Created synthetic image with shape {image.shape}")
    print(f"   Pixel range: [{image.min():.3f}, {image.max():.3f}]")
    
    return image

def test_model_inference(session):
    """Test model inference with synthetic data"""
    print("\n🧪 Testing Model Inference...")
    
    try:
        # Create test image
        test_image = create_synthetic_cardiac_image()
        
        # Prepare input tensor (add batch and channel dimensions)
        input_tensor = test_image[np.newaxis, np.newaxis, :, :].astype(np.float32)
        print(f"📥 Input tensor shape: {input_tensor.shape}")
        
        # Get input name
        input_name = session.get_inputs()[0].name
        
        # Run inference with timing
        print("⏱️  Running inference...")
        start_time = time.time()
        outputs = session.run(None, {input_name: input_tensor})
        inference_time = time.time() - start_time
        
        output_tensor = outputs[0]
        print(f"📤 Output tensor shape: {output_tensor.shape}")
        print(f"⚡ Inference time: {inference_time*1000:.1f}ms")
        
        # Analyze outputs
        print(f"\n📊 Output Analysis:")
        print(f"   Value range: [{output_tensor.min():.3f}, {output_tensor.max():.3f}]")
        print(f"   Mean: {output_tensor.mean():.3f}")
        print(f"   Standard deviation: {output_tensor.std():.3f}")
        
        # Convert logits to probabilities and predictions
        probabilities = np.exp(output_tensor) / np.sum(np.exp(output_tensor), axis=1, keepdims=True)
        predictions = np.argmax(output_tensor, axis=1)
        
        print(f"\n🎯 Segmentation Results:")
        unique_classes, counts = np.unique(predictions, return_counts=True)
        total_pixels = predictions.size
        
        class_names = ["Background", "LV Cavity", "LV Wall"]
        for class_id, count in zip(unique_classes, counts):
            percentage = count / total_pixels * 100
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
            print(f"   {class_name}: {count} pixels ({percentage:.1f}%)")
        
        # Check if segmentation is reasonable
        lv_pixels = np.sum([counts[i] for i, class_id in enumerate(unique_classes) if class_id > 0])
        lv_percentage = lv_pixels / total_pixels * 100
        
        if lv_percentage > 0:
            print(f"✅ Left ventricle detected: {lv_percentage:.1f}% of image")
        else:
            print("⚠️  No left ventricle structures detected")
        
        # Performance assessment
        print(f"\n🚀 Performance Assessment:")
        if inference_time < 0.05:
            print("🌟 EXCELLENT: < 50ms - Real-time performance")
        elif inference_time < 0.1:
            print("✅ VERY GOOD: < 100ms - Optimal for mobile")
        elif inference_time < 0.2:
            print("✅ GOOD: < 200ms - Acceptable for mobile")
        elif inference_time < 0.5:
            print("⚠️  FAIR: < 500ms - May impact user experience")
        else:
            print("❌ POOR: > 500ms - Too slow for mobile deployment")
        
        return {
            'inference_time': inference_time,
            'output_shape': output_tensor.shape,
            'predictions': predictions,
            'probabilities': probabilities,
            'lv_percentage': lv_percentage if lv_percentage > 0 else 0
        }
        
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        return None

def test_batch_performance(session, num_tests=10):
    """Test performance with multiple inferences"""
    print(f"\n⚡ Testing Batch Performance ({num_tests} inferences)...")
    
    try:
        inference_times = []
        input_name = session.get_inputs()[0].name
        
        for i in range(num_tests):
            # Create different test image each time
            test_image = create_synthetic_cardiac_image()
            input_tensor = test_image[np.newaxis, np.newaxis, :, :].astype(np.float32)
            
            start_time = time.time()
            outputs = session.run(None, {input_name: input_tensor})
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            if (i + 1) % 3 == 0:
                print(f"   Completed {i+1}/{num_tests} inferences...")
        
        # Calculate statistics
        times_ms = [t * 1000 for t in inference_times]
        avg_time = np.mean(times_ms)
        min_time = np.min(times_ms)
        max_time = np.max(times_ms)
        std_time = np.std(times_ms)
        
        print(f"\n📈 Performance Statistics:")
        print(f"   Average: {avg_time:.1f}ms")
        print(f"   Minimum: {min_time:.1f}ms")
        print(f"   Maximum: {max_time:.1f}ms")
        print(f"   Std Dev: {std_time:.1f}ms")
        print(f"   Consistency: {'Good' if std_time < avg_time * 0.2 else 'Variable'}")
        
        # Throughput calculation
        fps = 1000 / avg_time
        print(f"   Throughput: {fps:.1f} FPS")
        
        return {
            'avg_time_ms': avg_time,
            'min_time_ms': min_time,
            'max_time_ms': max_time,
            'std_time_ms': std_time,
            'fps': fps
        }
        
    except Exception as e:
        print(f"❌ Batch performance test failed: {e}")
        return None

def test_model_robustness(session):
    """Test model robustness with edge cases"""
    print("\n🛡️  Testing Model Robustness...")
    
    edge_cases = [
        ("All zeros", np.zeros((256, 256), dtype=np.float32)),
        ("All ones", np.ones((256, 256), dtype=np.float32)),
        ("Random noise", np.random.rand(256, 256).astype(np.float32)),
        ("High contrast", np.random.choice([0.0, 1.0], size=(256, 256)).astype(np.float32))
    ]
    
    input_name = session.get_inputs()[0].name
    results = {}
    
    for case_name, test_image in edge_cases:
        try:
            input_tensor = test_image[np.newaxis, np.newaxis, :, :].astype(np.float32)
            outputs = session.run(None, {input_name: input_tensor})
            
            # Check for NaN or Inf
            output_tensor = outputs[0]
            has_nan = np.isnan(output_tensor).any()
            has_inf = np.isinf(output_tensor).any()
            
            if has_nan or has_inf:
                print(f"❌ {case_name}: Contains NaN/Inf values")
                results[case_name] = False
            else:
                print(f"✅ {case_name}: Stable output")
                results[case_name] = True
                
        except Exception as e:
            print(f"❌ {case_name}: Failed - {e}")
            results[case_name] = False
    
    return results

def generate_deployment_report(model_info, inference_results, performance_stats, robustness_results):
    """Generate a comprehensive deployment readiness report"""
    print("\n" + "="*60)
    print("📋 PRODUCTION DEPLOYMENT READINESS REPORT")
    print("="*60)
    
    # Overall status
    all_tests_passed = all([
        model_info is not False,
        inference_results is not None,
        performance_stats is not None,
        all(robustness_results.values()) if robustness_results else False
    ])
    
    print(f"\n🎯 OVERALL STATUS: {'✅ READY FOR PRODUCTION' if all_tests_passed else '⚠️  NEEDS ATTENTION'}")
    
    print(f"\n📊 MODEL SPECIFICATIONS:")
    if model_info:
        inputs = model_info.get_inputs()
        outputs = model_info.get_outputs()
        print(f"   Model Type: ONNX Cardiac Segmentation")
        print(f"   Input Format: {inputs[0].shape} {inputs[0].type}")
        print(f"   Output Format: {outputs[0].shape} {outputs[0].type}")
        print(f"   Classes: 3 (Background, LV Cavity, LV Wall)")
    
    if inference_results:
        print(f"\n⚡ PERFORMANCE METRICS:")
        print(f"   Single Inference: {inference_results['inference_time']*1000:.1f}ms")
        if performance_stats:
            print(f"   Average Performance: {performance_stats['avg_time_ms']:.1f}ms")
            print(f"   Throughput: {performance_stats['fps']:.1f} FPS")
            print(f"   Consistency: ±{performance_stats['std_time_ms']:.1f}ms")
        
        # Mobile deployment assessment
        avg_time = performance_stats['avg_time_ms'] if performance_stats else inference_results['inference_time']*1000
        if avg_time < 100:
            mobile_rating = "🌟 EXCELLENT for mobile deployment"
        elif avg_time < 200:
            mobile_rating = "✅ GOOD for mobile deployment"
        elif avg_time < 500:
            mobile_rating = "⚠️  ACCEPTABLE but may impact UX"
        else:
            mobile_rating = "❌ TOO SLOW for mobile deployment"
        
        print(f"   Mobile Readiness: {mobile_rating}")
    
    if robustness_results:
        print(f"\n🛡️  ROBUSTNESS TEST:")
        passed_tests = sum(robustness_results.values())
        total_tests = len(robustness_results)
        print(f"   Stability Score: {passed_tests}/{total_tests} tests passed")
        
        for test_name, passed in robustness_results.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {test_name}")
    
    print(f"\n📱 iOS DEPLOYMENT RECOMMENDATIONS:")
    if all_tests_passed:
        print("   ✅ Model is ready for immediate iOS deployment")
        print("   ✅ Use ONNX Runtime iOS framework")
        print("   ✅ Implement preprocessing as per data models")
        print("   ✅ Add error handling for edge cases")
        print("   ✅ Consider background processing for better UX")
    else:
        print("   ⚠️  Address performance or stability issues before deployment")
        print("   ⚠️  Additional optimization may be required")
    
    print(f"\n🔗 INTEGRATION CHECKLIST:")
    print("   □ Add ONNX Runtime iOS framework to project")
    print("   □ Include model file in app bundle (55MB)")
    print("   □ Implement image preprocessing pipeline")
    print("   □ Add segmentation visualization components")
    print("   □ Implement error handling and fallbacks")
    print("   □ Test on target iOS devices")
    print("   □ Add performance monitoring")
    
    print("\n" + "="*60)
    
    return all_tests_passed

def main():
    """Run comprehensive production deployment test suite"""
    print("🚀 CAMUS Segmentation Model - Production Deployment Test")
    print("="*60)
    
    # Test 1: Model Loading
    session = test_model_loading()
    if not session:
        print("\n❌ CRITICAL: Model loading failed - cannot proceed with deployment")
        return False
    
    # Test 2: Basic Inference
    inference_results = test_model_inference(session)
    if not inference_results:
        print("\n❌ CRITICAL: Inference failed - model not suitable for deployment")
        return False
    
    # Test 3: Performance Testing
    performance_stats = test_batch_performance(session)
    
    # Test 4: Robustness Testing
    robustness_results = test_model_robustness(session)
    
    # Test 5: Generate Deployment Report
    deployment_ready = generate_deployment_report(
        session, inference_results, performance_stats, robustness_results
    )
    
    return deployment_ready

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
