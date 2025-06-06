# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a complete iOS application that performs real-time AI-powered cardiac left ventricle segmentation on 4-chamber ultrasound videos. The app allows users to upload ultrasound videos, processes them frame-by-frame using a 55MB ONNX model, and displays interactive segmentation results with multiple viewing modes.

## Key Commands

### Dependencies and Setup
```bash
# Install CocoaPods dependencies
pod install

# Open workspace (REQUIRED - not .xcodeproj)
open CAMUSTestApp.xcworkspace
```

### Building and Running
- **Always use `.xcworkspace`** - never open `.xcodeproj` after running `pod install`
- Build and run in Xcode: `⌘+R`
- Target iOS 15.0+ devices or simulators
- Recommended simulator: iPhone 14 or later for optimal performance

### Model File Management
- Core model: `camus_segmentation_real_weights.onnx` (55MB)
- Model must be included in Xcode target for bundle access
- Input: [1, 1, 256, 256] grayscale images
- Output: [1, 3, 256, 256] 3-class segmentation (background, LV cavity, LV wall)

## Architecture Overview

### Core Processing Pipeline
1. **Video Input** (`ContentView.swift`) - File selection, validation, and UI state management
2. **Frame Extraction** (`UltrasoundSegmentationProcessor.swift`) - AVFoundation-based frame extraction at 5 FPS
3. **AI Inference** (`CAMUSSegmentationModel.swift`) - ONNX Runtime model execution with CoreML optimization
4. **Results Display** (`SegmentationResultsView.swift`) - Interactive playback with multiple view modes

### Key Components

**UltrasoundSegmentationProcessor**: 
- Orchestrates the entire processing pipeline
- Extracts frames from video at 5 FPS using AVAssetImageGenerator
- Manages async/await processing with progress tracking
- Handles background processing without blocking UI

**CAMUSSegmentationModel**:
- Wraps ONNX Runtime for model inference
- Handles image preprocessing (resize to 256x256, grayscale conversion, normalization)
- Implements CoreML execution provider for device optimization
- Provides both async and sync inference methods

**SegmentationDataModels**:
- Contains data structures for segmentation results and statistics
- Includes visualization utilities for overlay creation
- Performance monitoring and optimization suggestions

### Data Flow
1. User selects video → ContentView validates duration/format
2. Video processing starts → frames extracted sequentially
3. Each frame → preprocessed → model inference → visualization created
4. Results collected → displayed in SegmentationResultsView with interactive controls

## Framework Dependencies

- **SwiftUI**: Declarative UI framework for all views
- **AVFoundation**: Video processing and frame extraction
- **onnxruntime-objc** (v1.16.0): AI model inference engine
- **Combine**: Reactive state management for processing updates

## Performance Characteristics

- **Model inference**: 60-80ms per frame on modern devices
- **Frame extraction**: ~10ms per frame
- **Total processing**: ~90ms per frame
- **Memory usage**: <100MB peak during processing
- **Video support**: .mp4, .mov formats, 0.5-30 second duration

## Common Development Patterns

### Error Handling
- All video processing errors are captured and displayed to user
- Model errors include recovery suggestions (ModelError enum)
- Validation occurs before processing (duration, format, tracks)

### Memory Management
- Frames processed individually to minimize memory footprint
- Background processing with proper actor isolation
- Model session cleanup in deinit

### UI State Management
- ObservableObject pattern for processing state
- Combine publishers for reactive UI updates
- Progress tracking with detailed status messages

## Testing Considerations

- Test with various video formats and qualities
- Verify performance on older iOS devices (minimum iOS 15.0)
- Validate memory usage with longer videos
- Test model accuracy with clinical ultrasound data
- Use UIImage.createTestUltrasoundImage() for development testing