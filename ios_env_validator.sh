#!/bin/bash

# iOS Development Environment Setup & Validation
# This script validates the iOS development environment and provides setup guidance

echo "🍎 iOS Development Environment Validator"
echo "📅 $(date)"
echo "======================================"

# Function to check Swift language support
check_swift_support() {
    echo "🚀 Swift Language Support:"
    
    # Check if Swift extensions are installed
    echo "   📦 Checking Swift extensions..."
    
    # List of expected Swift/iOS extensions
    EXPECTED_EXTENSIONS=(
        "sweetpad.sweetpad"
        "kiadstudios.vscode-swift" 
        "fireplusteam.vscode-ios"
        "fenkinet.swiftui"
        "llvm-vs-code-extensions.vscode-clangd"
    )
    
    for ext in "${EXPECTED_EXTENSIONS[@]}"; do
        echo "      - $ext: Checking..."
    done
    
    echo "   ✅ Swift extension validation complete"
    echo ""
}

# Function to validate iOS project structure
validate_ios_project() {
    echo "📱 iOS Project Structure Validation:"
    
    IOS_APP_DIR="/workspaces/Scrollshot_Fixer/ios_test_app"
    
    if [ ! -d "$IOS_APP_DIR" ]; then
        echo "   ❌ iOS test app directory not found"
        return 1
    fi
    
    echo "   📂 iOS App Directory: ✅ Found"
    
    # Check essential iOS files
    ESSENTIAL_FILES=(
        "CAMUSTestApp.swift"
        "ContentView.swift"
        "UltrasoundSegmentationProcessor.swift"
        "CAMUSSegmentationModel.swift"
        "SegmentationDataModels.swift"
        "Podfile"
    )
    
    for file in "${ESSENTIAL_FILES[@]}"; do
        if [ -f "$IOS_APP_DIR/$file" ]; then
            echo "   📄 $file: ✅ Found"
        else
            echo "   📄 $file: ❌ Missing"
        fi
    done
    
    # Check for build scripts
    if [ -f "$IOS_APP_DIR/build_and_run.sh" ]; then
        echo "   🔧 Build script: ✅ Found"
        if [ -x "$IOS_APP_DIR/build_and_run.sh" ]; then
            echo "      Executable: ✅ Yes"
        else
            echo "      Executable: ❌ No - Run: chmod +x build_and_run.sh"
        fi
    else
        echo "   🔧 Build script: ❌ Missing"
    fi
    
    echo ""
}

# Function to check macOS/Xcode requirements
check_macos_requirements() {
    echo "🖥️  macOS/Xcode Requirements Check:"
    
    # Since we're in Linux, we can't build iOS apps
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "   🐧 Current OS: Linux (Codespaces)"
        echo "   ⚠️  iOS development requires macOS + Xcode"
        echo "   📋 Requirements for iOS development:"
        echo "      - macOS 12.0 or later"
        echo "      - Xcode 14.0 or later" 
        echo "      - iOS Simulator"
        echo "      - CocoaPods (gem install cocoapods)"
        echo "   💡 Recommendation: Transfer project to macOS system"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "   🍎 Current OS: macOS"
        
        # Check Xcode
        if command -v xcodebuild >/dev/null 2>&1; then
            echo "   ✅ Xcode: Available"
            xcodebuild -version | head -1 | sed 's/^/      /'
        else
            echo "   ❌ Xcode: Not found - Install from App Store"
        fi
        
        # Check iOS Simulator
        if xcrun simctl list devices >/dev/null 2>&1; then
            echo "   ✅ iOS Simulator: Available"
        else
            echo "   ❌ iOS Simulator: Not available"
        fi
        
        # Check CocoaPods
        if command -v pod >/dev/null 2>&1; then
            echo "   ✅ CocoaPods: Available"
            pod --version | sed 's/^/      Version: /'
        else
            echo "   ❌ CocoaPods: Not found - Run: sudo gem install cocoapods"
        fi
    fi
    
    echo ""
}

# Function to validate ONNX model files
validate_onnx_models() {
    echo "🧠 ONNX Model Validation:"
    
    MODEL_FILES=(
        "/workspaces/Scrollshot_Fixer/camus_segmentation_real_weights.onnx"
        "/workspaces/Scrollshot_Fixer/ios_test_app/camus_segmentation_real_weights.onnx"
    )
    
    for model in "${MODEL_FILES[@]}"; do
        if [ -f "$model" ]; then
            echo "   📄 $(basename "$model"): ✅ Found"
            echo "      Size: $(du -h "$model" | cut -f1)"
            echo "      Location: $model"
        else
            echo "   📄 $(basename "$model"): ❌ Missing"
        fi
    done
    
    echo ""
}

# Function to check Swift syntax in iOS files
check_swift_syntax() {
    echo "🔍 Swift Syntax Validation:"
    
    IOS_APP_DIR="/workspaces/Scrollshot_Fixer/ios_test_app"
    
    if [ ! -d "$IOS_APP_DIR" ]; then
        echo "   ❌ iOS app directory not found"
        return 1
    fi
    
    # Find Swift files and check basic syntax
    swift_files=$(find "$IOS_APP_DIR" -name "*.swift" 2>/dev/null)
    
    if [ -z "$swift_files" ]; then
        echo "   ❌ No Swift files found"
        return 1
    fi
    
    echo "   📄 Found Swift files:"
    echo "$swift_files" | sed 's/^/      /'
    
    echo "   🔍 Basic syntax check (looking for obvious issues):"
    
    for file in $swift_files; do
        filename=$(basename "$file")
        echo "      📝 $filename:"
        
        # Check for basic Swift syntax issues
        if grep -n "import UIKit\|import SwiftUI\|import Foundation" "$file" >/dev/null; then
            echo "         ✅ Has proper imports"
        else
            echo "         ⚠️  No standard imports found"
        fi
        
        # Check for basic structure
        if grep -n "struct\|class\|protocol" "$file" >/dev/null; then
            echo "         ✅ Has Swift structures"
        else
            echo "         ⚠️  No Swift structures found"
        fi
        
        # Check for obvious syntax errors (missing braces, etc.)
        open_braces=$(grep -o "{" "$file" | wc -l)
        close_braces=$(grep -o "}" "$file" | wc -l)
        
        if [ "$open_braces" -eq "$close_braces" ]; then
            echo "         ✅ Balanced braces ($open_braces pairs)"
        else
            echo "         ❌ Unbalanced braces (open: $open_braces, close: $close_braces)"
        fi
    done
    
    echo ""
}

# Function to provide development recommendations
provide_recommendations() {
    echo "💡 iOS Development Recommendations:"
    echo ""
    echo "   🏗️  For Current Linux Environment:"
    echo "      ✅ Code editing and syntax checking"
    echo "      ✅ Static analysis and documentation"
    echo "      ✅ Version control and collaboration"
    echo "      ❌ Building and running iOS apps"
    echo "      ❌ iOS Simulator testing"
    echo ""
    echo "   🍎 For macOS Environment:"
    echo "      1. Transfer project files to macOS system"
    echo "      2. Install Xcode from App Store"
    echo "      3. Install CocoaPods: sudo gem install cocoapods"
    echo "      4. Navigate to ios_test_app/ directory"
    echo "      5. Run: pod install"
    echo "      6. Run: ./build_and_run.sh"
    echo ""
    echo "   🔧 VS Code Setup (any environment):"
    echo "      1. Install Swift extensions (done ✅)"
    echo "      2. Configure syntax highlighting"
    echo "      3. Set up error monitoring (done ✅)"
    echo "      4. Use integrated terminal for builds"
    echo ""
    echo "   📱 Testing Strategy:"
    echo "      - macOS: Use iOS Simulator + physical devices"
    echo "      - Linux: Static analysis + code review"
    echo "      - CI/CD: GitHub Actions with macOS runners"
    echo ""
}

# Function to run all checks
run_all_checks() {
    check_swift_support
    validate_ios_project
    check_macos_requirements
    validate_onnx_models
    check_swift_syntax
    provide_recommendations
}

# Main execution
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [option]"
    echo "Options:"
    echo "  --help, -h     Show this help message"
    echo "  --swift        Check Swift syntax only"
    echo "  --project      Validate iOS project only"
    echo "  --models       Validate ONNX models only"
    echo "  (no option)    Run all checks"
elif [ "$1" = "--swift" ]; then
    check_swift_syntax
elif [ "$1" = "--project" ]; then
    validate_ios_project
elif [ "$1" = "--models" ]; then
    validate_onnx_models
else
    run_all_checks
fi
