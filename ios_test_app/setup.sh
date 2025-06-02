#!/bin/bash

# CAMUS iOS Test App Setup Script
# This script copies the necessary files from the main implementation to the test app

echo "🚀 Setting up CAMUS iOS Test App..."
echo "=" * 50

# Define source and destination paths
SOURCE_DIR="/workspaces/Scrollshot_Fixer/ios_implementation"
DEST_DIR="/workspaces/Scrollshot_Fixer/ios_test_app"
MODEL_FILE="/workspaces/Scrollshot_Fixer/camus_segmentation_real_weights.onnx"

# Check if source files exist
if [ ! -f "$SOURCE_DIR/CAMUSSegmentationModel.swift" ]; then
    echo "❌ Error: CAMUSSegmentationModel.swift not found in $SOURCE_DIR"
    exit 1
fi

if [ ! -f "$SOURCE_DIR/SegmentationDataModels.swift" ]; then
    echo "❌ Error: SegmentationDataModels.swift not found in $SOURCE_DIR"
    exit 1
fi

if [ ! -f "$MODEL_FILE" ]; then
    echo "❌ Error: ONNX model file not found at $MODEL_FILE"
    exit 1
fi

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

echo "📱 Copying core segmentation files..."

# Copy the core segmentation model and data structures
cp "$SOURCE_DIR/CAMUSSegmentationModel.swift" "$DEST_DIR/"
echo "✅ Copied CAMUSSegmentationModel.swift"

cp "$SOURCE_DIR/SegmentationDataModels.swift" "$DEST_DIR/"
echo "✅ Copied SegmentationDataModels.swift"

# Copy the ONNX model (this will be large - 55MB)
echo "📦 Copying ONNX model file (55MB)..."
cp "$MODEL_FILE" "$DEST_DIR/"
echo "✅ Copied camus_segmentation_real_weights.onnx"

# Check file sizes
echo ""
echo "📊 File sizes:"
echo "CAMUSSegmentationModel.swift: $(du -h "$DEST_DIR/CAMUSSegmentationModel.swift" | cut -f1)"
echo "SegmentationDataModels.swift: $(du -h "$DEST_DIR/SegmentationDataModels.swift" | cut -f1)"
echo "ONNX Model: $(du -h "$DEST_DIR/camus_segmentation_real_weights.onnx" | cut -f1)"

# Count total lines of Swift code in the test app
echo ""
echo "📝 Test app structure:"
SWIFT_FILES=$(find "$DEST_DIR" -name "*.swift" 2>/dev/null)
TOTAL_LINES=0

for file in $SWIFT_FILES; do
    if [ -f "$file" ]; then
        lines=$(wc -l < "$file")
        TOTAL_LINES=$((TOTAL_LINES + lines))
        echo "$(basename "$file"): $lines lines"
    fi
done

echo "Total Swift code: $TOTAL_LINES lines"

echo ""
echo "🎯 Next steps:"
echo "1. cd $DEST_DIR"
echo "2. pod install"
echo "3. open CAMUSTestApp.xcworkspace"
echo "4. Build and run the app"
echo ""
echo "✅ Setup complete! Your iOS test app is ready for development."
