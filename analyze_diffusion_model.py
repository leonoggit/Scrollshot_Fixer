#!/usr/bin/env python3
"""
CAMUS Diffusion Model Analysis and ONNX Conversion
Following the successful methodology used for segmentation model conversion
"""

import torch
import sys
import os
from collections import OrderedDict

# Add echogains to Python path for diffusion model imports
sys.path.insert(0, '/workspaces/Scrollshot_Fixer/echogains')

def analyze_diffusion_checkpoint():
    """Analyze the diffusion model checkpoint structure"""
    
    print("=== CAMUS Diffusion Model Analysis ===")
    
    # Load the diffusion model checkpoint
    model_path = "/workspaces/Scrollshot_Fixer/CAMUS_diffusion_model.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None
    
    print(f"📂 Loading checkpoint from: {model_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        print(f"✅ Checkpoint loaded successfully")
        
        # Analyze checkpoint structure
        print(f"\n📊 Checkpoint Structure:")
        if isinstance(checkpoint, dict):
            for key in checkpoint.keys():
                value = checkpoint[key]
                if isinstance(value, torch.Tensor):
                    print(f"  📝 {key}: Tensor {value.shape} ({value.dtype})")
                elif isinstance(value, dict):
                    print(f"  📁 {key}: Dict with {len(value)} keys")
                    if len(value) < 20:  # Show keys if not too many
                        for subkey in list(value.keys())[:10]:
                            subvalue = value[subkey]
                            if isinstance(subvalue, torch.Tensor):
                                print(f"    📝 {subkey}: Tensor {subvalue.shape}")
                            else:
                                print(f"    📄 {subkey}: {type(subvalue)}")
                        if len(value) > 10:
                            print(f"    ... and {len(value) - 10} more keys")
                else:
                    print(f"  📄 {key}: {type(value)} - {str(value)[:100]}")
        
        # Try to identify the model state dict
        state_dict = None
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"\n✅ Found 'state_dict' key with {len(state_dict)} parameters")
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"\n✅ Found 'model_state_dict' key with {len(state_dict)} parameters")
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
            print(f"\n✅ Found 'model' key with {len(state_dict)} parameters")
        elif isinstance(checkpoint, dict) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            state_dict = checkpoint
            print(f"\n✅ Checkpoint appears to be state_dict directly with {len(state_dict)} parameters")
        
        if state_dict is None:
            print("❌ Could not identify state_dict in checkpoint")
            return checkpoint
        
        # Analyze state dict structure
        print(f"\n🔍 Model Architecture Analysis:")
        print(f"Total parameters: {len(state_dict)}")
        
        # Group parameters by module
        modules = {}
        for param_name in state_dict.keys():
            # Extract module name (everything before the last dot)
            if '.' in param_name:
                module_name = '.'.join(param_name.split('.')[:-1])
            else:
                module_name = 'root'
            
            if module_name not in modules:
                modules[module_name] = []
            modules[module_name].append(param_name)
        
        print(f"\n📊 Module Structure:")
        for module_name, params in modules.items():
            print(f"  🔧 {module_name}: {len(params)} parameters")
            # Show first few parameters for each module
            for param in params[:3]:
                tensor = state_dict[param]
                print(f"    📝 {param}: {tensor.shape} ({tensor.dtype})")
            if len(params) > 3:
                print(f"    ... and {len(params) - 3} more parameters")
        
        # Look for common diffusion model patterns
        print(f"\n🔍 Diffusion Model Pattern Analysis:")
        
        # Check for UNet patterns
        unet_patterns = ['input_blocks', 'middle_block', 'output_blocks', 'out', 'time_embed']
        found_patterns = []
        for pattern in unet_patterns:
            matching_keys = [k for k in state_dict.keys() if pattern in k]
            if matching_keys:
                found_patterns.append(pattern)
                print(f"  ✅ Found {pattern}: {len(matching_keys)} parameters")
        
        if found_patterns:
            print(f"  🎯 Detected UNet architecture with patterns: {found_patterns}")
        else:
            print(f"  ❓ No standard UNet patterns detected")
        
        # Check for attention patterns
        attention_keys = [k for k in state_dict.keys() if 'attn' in k.lower() or 'attention' in k.lower()]
        if attention_keys:
            print(f"  🧠 Found attention mechanisms: {len(attention_keys)} parameters")
        
        # Check for time embedding patterns
        time_keys = [k for k in state_dict.keys() if 'time' in k.lower() or 'timestep' in k.lower()]
        if time_keys:
            print(f"  ⏰ Found time embeddings: {len(time_keys)} parameters")
        
        # Analyze tensor shapes for input/output dimensions
        print(f"\n📐 Key Tensor Shapes:")
        for param_name, tensor in list(state_dict.items())[:10]:
            print(f"  📝 {param_name}: {tensor.shape}")
        
        return checkpoint
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None

def analyze_repaint_architecture():
    """Analyze the RePaint diffusion architecture from the codebase"""
    
    print(f"\n=== RePaint Architecture Analysis ===")
    
    # Check if guided_diffusion modules exist
    guided_diffusion_path = "/workspaces/Scrollshot_Fixer/echogains2/RePaint/guided_diffusion"
    
    if not os.path.exists(guided_diffusion_path):
        print(f"❌ guided_diffusion path not found: {guided_diffusion_path}")
        return
    
    print(f"✅ Found guided_diffusion module at: {guided_diffusion_path}")
    
    # List available modules
    modules = []
    for item in os.listdir(guided_diffusion_path):
        if item.endswith('.py') and item != '__init__.py':
            modules.append(item[:-3])  # Remove .py extension
    
    print(f"📊 Available modules: {modules}")
    
    # Key modules we expect for diffusion models
    expected_modules = ['unet', 'gaussian_diffusion', 'script_util']
    for module in expected_modules:
        if module in modules:
            print(f"  ✅ Found {module}.py")
        else:
            print(f"  ❌ Missing {module}.py")

def analyze_echogains_structure():
    """Analyze the echogains package to understand diffusion model implementation"""
    print("\n=== Echogains Package Analysis ===")
    
    echogains_path = "/workspaces/Scrollshot_Fixer/echogains"
    
    if not os.path.exists(echogains_path):
        print(f"❌ Echogains path not found: {echogains_path}")
        return
    
    print(f"📂 Echogains path: {echogains_path}")
    
    # List all Python files
    python_files = []
    for root, dirs, files in os.walk(echogains_path):
        for file in files:
            if file.endswith('.py'):
                rel_path = os.path.relpath(os.path.join(root, file), echogains_path)
                python_files.append(rel_path)
    
    print(f"🐍 Python files found: {len(python_files)}")
    for file in sorted(python_files)[:10]:  # Show first 10
        print(f"   {file}")
    if len(python_files) > 10:
        print(f"   ... and {len(python_files) - 10} more")
    
    # Look for key diffusion files
    key_files = ['inference.py', 'utils.py', 'CONST.py']
    print(f"\n🔍 Key files analysis:")
    for file in key_files:
        file_path = os.path.join(echogains_path, file)
        if os.path.exists(file_path):
            print(f"✅ {file} found")
            # Check file size
            size = os.path.getsize(file_path)
            print(f"   Size: {size} bytes")
        else:
            print(f"❌ {file} not found")
    
    # Try to import echogains
    try:
        import echogains
        print(f"\n✅ Successfully imported echogains")
        
        # Check available attributes
        attrs = [attr for attr in dir(echogains) if not attr.startswith('_')]
        print(f"   Available attributes: {attrs}")
        
        # Try specific imports
        modules_to_check = ['inference', 'utils', 'CONST']
        for module in modules_to_check:
            try:
                mod = getattr(echogains, module, None)
                if mod:
                    print(f"✅ echogains.{module} available")
                else:
                    exec(f"from echogains import {module}")
                    print(f"✅ echogains.{module} imported successfully")
            except Exception as e:
                print(f"❌ echogains.{module} failed: {e}")
                
    except Exception as e:
        print(f"❌ Failed to import echogains: {e}")

def look_for_diffusion_architecture():
    """Search for diffusion model architecture in echogains"""
    print("\n=== Diffusion Model Architecture Search ===")
    
    # Check for RePaint directory
    repaint_path = "/workspaces/Scrollshot_Fixer/echogains/RePaint"
    if os.path.exists(repaint_path):
        print(f"✅ RePaint directory found: {repaint_path}")
        
        # List contents
        try:
            contents = os.listdir(repaint_path)
            print(f"   Contents: {contents}")
            
            # Look for guided_diffusion
            guided_diff_path = os.path.join(repaint_path, "guided_diffusion")
            if os.path.exists(guided_diff_path):
                print(f"✅ guided_diffusion found: {guided_diff_path}")
                guided_contents = os.listdir(guided_diff_path)
                print(f"   guided_diffusion contents: {guided_contents[:10]}...")
        except Exception as e:
            print(f"❌ Error listing RePaint contents: {e}")
    else:
        print(f"❌ RePaint directory not found: {repaint_path}")
    
    # Search for model-related files
    search_terms = ['model', 'diffusion', 'unet', 'net']
    echogains_path = "/workspaces/Scrollshot_Fixer/echogains"
    
    for root, dirs, files in os.walk(echogains_path):
        for file in files:
            if file.endswith('.py'):
                for term in search_terms:
                    if term in file.lower():
                        rel_path = os.path.relpath(os.path.join(root, file), echogains_path)
                        print(f"🎯 Found {term}-related file: {rel_path}")

if __name__ == "__main__":
    print("Starting CAMUS Diffusion Model Analysis...")
    
    # Analyze the checkpoint
    checkpoint = analyze_diffusion_checkpoint()
    
    # Analyze the codebase architecture
    analyze_repaint_architecture()
    
    # Analyze the echogains structure
    analyze_echogains_structure()
    
    # Look for diffusion architecture in echogains
    look_for_diffusion_architecture()
    
    print(f"\n✅ Analysis complete!")
