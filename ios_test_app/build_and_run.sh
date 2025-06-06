#!/bin/bash

# CAMUS iOS Test App Build Script
# Run this on macOS with Xcode installed

set -e

echo "🏗️  CAMUS iOS Test App Build Script"
echo "====================================="

# Check prerequisites
echo "📋 Checking prerequisites..."

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ Error: This script requires macOS with Xcode"
    echo "   Current OS: $OSTYPE"
    exit 1
fi

# Check if Xcode is installed
if ! command -v xcodebuild &> /dev/null; then
    echo "❌ Error: Xcode command line tools not found"
    echo "   Install with: xcode-select --install"
    exit 1
fi

# Check Xcode version
XCODE_VERSION=$(xcodebuild -version | head -n 1 | sed 's/Xcode //')
echo "✅ Xcode version: $XCODE_VERSION"

# Check if CocoaPods is installed
if ! command -v pod &> /dev/null; then
    echo "⚠️  CocoaPods not found. Installing..."
    sudo gem install cocoapods
fi

POD_VERSION=$(pod --version)
echo "✅ CocoaPods version: $POD_VERSION"

# Navigate to project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "📁 Project directory: $PROJECT_DIR"

# Check if ONNX model file exists
if [[ ! -f "$PROJECT_DIR/camus_segmentation_real_weights.onnx" ]]; then
    echo "❌ Error: ONNX model file not found"
    echo "   Expected: $PROJECT_DIR/camus_segmentation_real_weights.onnx"
    echo "   Please copy the model file to the ios_test_app directory"
    exit 1
fi

MODEL_SIZE=$(ls -lh "$PROJECT_DIR/camus_segmentation_real_weights.onnx" | awk '{print $5}')
echo "✅ ONNX model file found (size: $MODEL_SIZE)"

# Install CocoaPods dependencies
echo ""
echo "📦 Installing CocoaPods dependencies..."
pod install --verbose

if [[ $? -ne 0 ]]; then
    echo "❌ Error: CocoaPods installation failed"
    echo "   Try: pod repo update && pod install"
    exit 1
fi

echo "✅ CocoaPods dependencies installed successfully"

# Check if workspace was created
if [[ ! -f "$PROJECT_DIR/CAMUSTestApp.xcworkspace" ]]; then
    echo "❌ Error: Xcode workspace not created"
    echo "   CocoaPods should have created CAMUSTestApp.xcworkspace"
    exit 1
fi

echo "✅ Xcode workspace created"

# List available simulators
echo ""
echo "📱 Available iOS Simulators:"
xcrun simctl list devices available | grep "iPhone"

# Find iPhone 16 Pro simulator
SIMULATOR_ID=$(xcrun simctl list devices available | grep "iPhone 16 Pro (" | head -n 1 | grep -o "([^)]*)" | tr -d "()")

if [[ -z "$SIMULATOR_ID" ]]; then
    echo "⚠️  iPhone 16 Pro simulator not found. Looking for alternatives..."
    SIMULATOR_ID=$(xcrun simctl list devices available | grep -E "iPhone (15|14|13)" | head -n 1 | grep -o "([^)]*)" | tr -d "()")
    
    if [[ -z "$SIMULATOR_ID" ]]; then
        echo "❌ Error: No suitable iPhone simulator found"
        echo "   Please install iPhone simulators in Xcode"
        exit 1
    fi
fi

SIMULATOR_NAME=$(xcrun simctl list devices available | grep "$SIMULATOR_ID" | sed 's/.*iPhone/iPhone/' | sed 's/ (.*//')
echo "🎯 Target simulator: $SIMULATOR_NAME ($SIMULATOR_ID)"

# Build the project
echo ""
echo "🔨 Building CAMUSTestApp..."

xcodebuild -workspace CAMUSTestApp.xcworkspace \
           -scheme CAMUSTestApp \
           -destination "platform=iOS Simulator,id=$SIMULATOR_ID" \
           -configuration Debug \
           clean build

if [[ $? -ne 0 ]]; then
    echo "❌ Build failed"
    echo ""
    echo "🔍 Common solutions:"
    echo "   1. Open CAMUSTestApp.xcworkspace in Xcode"
    echo "   2. Check that all files are properly added to the target"
    echo "   3. Verify the ONNX model file is in the bundle resources"
    echo "   4. Check the deployment target is set to iOS 15.0+"
    exit 1
fi

echo "✅ Build successful!"

# Install and launch the app
echo ""
echo "🚀 Installing and launching app on simulator..."

# Boot the simulator if not already running
xcrun simctl boot "$SIMULATOR_ID" 2>/dev/null || true

# Install the app
APP_PATH=$(find ~/Library/Developer/Xcode/DerivedData -name "CAMUSTestApp.app" | head -n 1)
if [[ -z "$APP_PATH" ]]; then
    echo "❌ Error: Could not find built app"
    exit 1
fi

xcrun simctl install "$SIMULATOR_ID" "$APP_PATH"
echo "✅ App installed on simulator"

# Launch the app
BUNDLE_ID="com.example.CAMUSTestApp"  # Adjust based on your actual bundle ID
xcrun simctl launch "$SIMULATOR_ID" "$BUNDLE_ID"
echo "✅ App launched on simulator"

# Open simulator
open -a Simulator

echo ""
echo "🎉 CAMUS iOS Test App successfully built and launched!"
echo ""
echo "📖 Next steps:"
echo "   1. The app should now be running on the $SIMULATOR_NAME simulator"
echo "   2. Test video upload functionality"
echo "   3. Verify segmentation processing works"
echo "   4. Check for any runtime errors in Xcode console"
echo ""
echo "🔍 Troubleshooting:"
echo "   - If the app crashes, check Xcode console for error messages"
echo "   - If ONNX model fails to load, verify the file is in the app bundle"
echo "   - If segmentation fails, check ONNX Runtime version compatibility"
