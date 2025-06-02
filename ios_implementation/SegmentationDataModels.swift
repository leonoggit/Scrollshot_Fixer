import UIKit
import CoreGraphics
import Accelerate

// MARK: - UIImage Extensions for Preprocessing

extension UIImage {
    
    /**
     * Resize image to specified size using high-quality interpolation
     */
    func resized(to size: CGSize) -> UIImage? {
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1.0
        format.opaque = false
        
        let renderer = UIGraphicsImageRenderer(size: size, format: format)
        
        return renderer.image { _ in
            self.draw(in: CGRect(origin: .zero, size: size))
        }
    }
    
    /**
     * Extract grayscale pixel data as UInt8 array
     * Returns array of pixel values from 0-255
     */
    func grayscalePixelData() -> [UInt8]? {
        guard let cgImage = self.cgImage else {
            print("❌ Error: Could not get CGImage from UIImage")
            return nil
        }
        
        let width = cgImage.width
        let height = cgImage.height
        let bytesPerPixel = 1
        let bytesPerRow = bytesPerPixel * width
        let bitsPerComponent = 8
        
        var pixelData = [UInt8](repeating: 0, count: width * height)
        
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        )
        
        guard let context = context else {
            print("❌ Error: Could not create CGContext for grayscale conversion")
            return nil
        }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        return pixelData
    }
    
    /**
     * Create a test ultrasound-like image for development/testing
     */
    static func createTestUltrasoundImage(size: CGSize = CGSize(width: 256, height: 256)) -> UIImage {
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1.0
        format.opaque = true
        
        let renderer = UIGraphicsImageRenderer(size: size, format: format)
        
        return renderer.image { context in
            let rect = CGRect(origin: .zero, size: size)
            
            // Dark background (typical ultrasound)
            UIColor.black.setFill()
            context.fill(rect)
            
            // Add some circular patterns to simulate cardiac structures
            UIColor.gray.setFill()
            
            // Simulate left ventricle cavity (brighter area)
            let cavityCenter = CGPoint(x: size.width * 0.6, y: size.height * 0.5)
            let cavityRadius = min(size.width, size.height) * 0.15
            let cavityRect = CGRect(
                x: cavityCenter.x - cavityRadius,
                y: cavityCenter.y - cavityRadius,
                width: cavityRadius * 2,
                height: cavityRadius * 2
            )
            context.cgContext.fillEllipse(in: cavityRect)
            
            // Simulate left ventricle wall (medium gray)
            UIColor.darkGray.setFill()
            let wallRadius = cavityRadius * 1.5
            let wallRect = CGRect(
                x: cavityCenter.x - wallRadius,
                y: cavityCenter.y - wallRadius,
                width: wallRadius * 2,
                height: wallRadius * 2
            )
            context.cgContext.fillEllipse(in: wallRect)
            
            // Add some noise patterns typical in ultrasound
            for _ in 0..<100 {
                let x = CGFloat.random(in: 0...size.width)
                let y = CGFloat.random(in: 0...size.height)
                let brightness = CGFloat.random(in: 0.1...0.3)
                
                UIColor(white: brightness, alpha: 1.0).setFill()
                let noiseRect = CGRect(x: x, y: y, width: 2, height: 2)
                context.fill(noiseRect)
            }
        }
    }
}

// MARK: - Data Models

/**
 * Comprehensive segmentation result containing all inference outputs and metadata
 */
struct SegmentationResult {
    let segmentationMask: [[Int]]      // 2D array: 0=background, 1=LV cavity, 2=LV wall
    let confidence: Float              // Overall prediction confidence [0-1]
    let inferenceTime: Double          // Inference time in seconds
    let imageSize: CGSize             // Size of segmentation mask
    let segmentationStats: SegmentationStats // Detailed pixel statistics
    
    // MARK: - Convenience Properties
    
    /**
     * Whether any left ventricle structures were detected
     */
    var hasLeftVentricle: Bool {
        return segmentationStats.leftVentriclePixels > 0
    }
    
    /**
     * Percentage of image that is left ventricle
     */
    var leftVentriclePercentage: Float {
        guard segmentationStats.totalPixels > 0 else { return 0.0 }
        return Float(segmentationStats.leftVentriclePixels) / Float(segmentationStats.totalPixels) * 100.0
    }
    
    /**
     * Ratio of cavity to wall pixels (useful for cardiac assessment)
     */
    var cavityToWallRatio: Float {
        guard segmentationStats.wallPixels > 0 else { return 0.0 }
        return Float(segmentationStats.cavityPixels) / Float(segmentationStats.wallPixels)
    }
    
    /**
     * Quality assessment based on confidence and structure detection
     */
    var qualityScore: Float {
        let confidenceWeight: Float = 0.6
        let detectionWeight: Float = 0.4
        
        let confidenceScore = confidence
        let detectionScore: Float = hasLeftVentricle ? 1.0 : 0.0
        
        return confidenceScore * confidenceWeight + detectionScore * detectionWeight
    }
    
    /**
     * Human-readable summary of the segmentation
     */
    var summary: String {
        if !hasLeftVentricle {
            return "No left ventricle detected"
        }
        
        return """
        Left Ventricle Detected
        Confidence: \(String(format: "%.1f", confidence * 100))%
        Coverage: \(String(format: "%.1f", leftVentriclePercentage))%
        Cavity/Wall Ratio: \(String(format: "%.2f", cavityToWallRatio))
        Quality Score: \(String(format: "%.1f", qualityScore * 100))%
        """
    }
}

/**
 * Detailed pixel-level statistics from segmentation
 */
struct SegmentationStats {
    let backgroundPixels: Int      // Class 0 pixels
    let cavityPixels: Int         // Class 1 pixels (LV cavity)
    let wallPixels: Int           // Class 2 pixels (LV wall)
    let leftVentriclePixels: Int  // Combined LV pixels (cavity + wall)
    let totalPixels: Int          // Total pixels in image
    
    // MARK: - Convenience Properties
    
    var backgroundPercentage: Float {
        return Float(backgroundPixels) / Float(totalPixels) * 100.0
    }
    
    var cavityPercentage: Float {
        return Float(cavityPixels) / Float(totalPixels) * 100.0
    }
    
    var wallPercentage: Float {
        return Float(wallPixels) / Float(totalPixels) * 100.0
    }
    
    var leftVentriclePercentage: Float {
        return Float(leftVentriclePixels) / Float(totalPixels) * 100.0
    }
}

/**
 * Model-specific error types for better error handling
 */
enum ModelError: Error {
    case sessionNotInitialized
    case preprocessingFailed
    case outputProcessingFailed(String)
    case invalidInput
    case invalidModel(String)
    case unknownError(String)
    
    var localizedDescription: String {
        switch self {
        case .sessionNotInitialized:
            return "ONNX session not initialized. Check that the model file is in your app bundle."
        case .preprocessingFailed:
            return "Image preprocessing failed. Ensure the input is a valid UIImage."
        case .outputProcessingFailed(let details):
            return "Output processing failed: \(details)"
        case .invalidInput:
            return "Invalid input provided to the model."
        case .invalidModel(let details):
            return "Invalid model configuration: \(details)"
        case .unknownError(let details):
            return "Unknown error occurred: \(details)"
        }
    }
    
    var isRecoverable: Bool {
        switch self {
        case .sessionNotInitialized, .invalidModel:
            return false // These require app restart or fix
        case .preprocessingFailed, .outputProcessingFailed, .invalidInput, .unknownError:
            return true  // These might work with different input
        }
    }
}

// MARK: - Segmentation Visualizer

/**
 * Utility class for creating visual overlays and result images
 */
class SegmentationVisualizer {
    
    // Color scheme for segmentation classes
    static let classColors: [(red: UInt8, green: UInt8, blue: UInt8)] = [
        (0, 0, 0),       // Class 0: Background (transparent/black)
        (255, 0, 0),     // Class 1: LV Cavity (red)
        (0, 255, 0)      // Class 2: LV Wall (green)
    ]
    
    /**
     * Create a colored overlay image from segmentation mask
     */
    static func createOverlayImage(
        from segmentationMask: [[Int]], 
        originalSize: CGSize,
        alpha: Float = 0.6
    ) -> UIImage? {
        
        let maskHeight = segmentationMask.count
        let maskWidth = segmentationMask[0].count
        
        // Create RGBA color data
        var colorData = [UInt8]()
        colorData.reserveCapacity(maskHeight * maskWidth * 4)
        
        for h in 0..<maskHeight {
            for w in 0..<maskWidth {
                let classIndex = segmentationMask[h][w]
                let color = getColorForClass(classIndex)
                
                colorData.append(color.red)
                colorData.append(color.green)
                colorData.append(color.blue)
                colorData.append(UInt8(alpha * 255)) // Alpha channel
            }
        }
        
        // Create CGImage from color data
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        
        guard let context = CGContext(
            data: &colorData,
            width: maskWidth,
            height: maskHeight,
            bitsPerComponent: 8,
            bytesPerRow: maskWidth * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ) else {
            print("❌ Error: Could not create CGContext for overlay")
            return nil
        }
        
        guard let cgImage = context.makeImage() else {
            print("❌ Error: Could not create CGImage from context")
            return nil
        }
        
        let overlayImage = UIImage(cgImage: cgImage)
        
        // Resize to match original image size if needed
        if originalSize != CGSize(width: maskWidth, height: maskHeight) {
            return overlayImage.resized(to: originalSize)
        }
        
        return overlayImage
    }
    
    /**
     * Combine original image with segmentation overlay
     */
    static func combineImages(original: UIImage, overlay: UIImage, overlayAlpha: CGFloat = 0.6) -> UIImage? {
        let format = UIGraphicsImageRendererFormat()
        format.scale = original.scale
        format.opaque = false
        
        let renderer = UIGraphicsImageRenderer(size: original.size, format: format)
        
        return renderer.image { _ in
            // Draw original image
            original.draw(in: CGRect(origin: .zero, size: original.size))
            
            // Draw overlay with transparency
            overlay.draw(
                in: CGRect(origin: .zero, size: original.size), 
                blendMode: .normal, 
                alpha: overlayAlpha
            )
        }
    }
    
    /**
     * Create a side-by-side comparison image
     */
    static func createComparisonImage(original: UIImage, segmented: UIImage) -> UIImage? {
        let totalWidth = original.size.width * 2
        let height = original.size.height
        let totalSize = CGSize(width: totalWidth, height: height)
        
        let format = UIGraphicsImageRendererFormat()
        format.scale = original.scale
        format.opaque = true
        
        let renderer = UIGraphicsImageRenderer(size: totalSize, format: format)
        
        return renderer.image { context in
            // Draw original on left
            original.draw(in: CGRect(x: 0, y: 0, width: original.size.width, height: height))
            
            // Draw segmented on right
            segmented.draw(in: CGRect(x: original.size.width, y: 0, width: original.size.width, height: height))
            
            // Add separator line
            context.cgContext.setStrokeColor(UIColor.white.cgColor)
            context.cgContext.setLineWidth(2.0)
            context.cgContext.move(to: CGPoint(x: original.size.width, y: 0))
            context.cgContext.addLine(to: CGPoint(x: original.size.width, y: height))
            context.cgContext.strokePath()
        }
    }
    
    /**
     * Create a legend image showing class colors
     */
    static func createLegendImage(size: CGSize = CGSize(width: 200, height: 80)) -> UIImage {
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1.0
        format.opaque = true
        
        let renderer = UIGraphicsImageRenderer(size: size, format: format)
        
        return renderer.image { context in
            // White background
            UIColor.white.setFill()
            context.fill(CGRect(origin: .zero, size: size))
            
            let rect = CGRect(origin: .zero, size: size)
            let itemHeight = rect.height / 3
            
            // Draw legend items
            let legendItems = [
                ("Background", UIColor.black),
                ("LV Cavity", UIColor.red),
                ("LV Wall", UIColor.green)
            ]
            
            for (index, (label, color)) in legendItems.enumerated() {
                let y = CGFloat(index) * itemHeight
                
                // Color square
                color.setFill()
                let colorRect = CGRect(x: 10, y: y + itemHeight/4, width: itemHeight/2, height: itemHeight/2)
                context.fill(colorRect)
                
                // Label text
                let textRect = CGRect(x: 20 + itemHeight/2, y: y, width: rect.width - 30 - itemHeight/2, height: itemHeight)
                let attributes: [NSAttributedString.Key: Any] = [
                    .font: UIFont.systemFont(ofSize: 12),
                    .foregroundColor: UIColor.black
                ]
                label.draw(in: textRect, withAttributes: attributes)
            }
        }
    }
    
    // MARK: - Private Methods
    
    private static func getColorForClass(_ classIndex: Int) -> (red: UInt8, green: UInt8, blue: UInt8) {
        guard classIndex >= 0 && classIndex < classColors.count else {
            return (128, 128, 128) // Gray for unknown classes
        }
        return classColors[classIndex]
    }
}

// MARK: - Performance Monitoring

/**
 * Utility class for monitoring model performance and providing optimization suggestions
 */
class PerformanceMonitor {
    
    private static var inferenceTimes: [Double] = []
    private static let maxStoredTimes = 100
    
    /**
     * Record an inference time for performance tracking
     */
    static func recordInferenceTime(_ time: Double) {
        inferenceTimes.append(time)
        
        // Keep only recent times
        if inferenceTimes.count > maxStoredTimes {
            inferenceTimes.removeFirst()
        }
    }
    
    /**
     * Get performance statistics
     */
    static func getPerformanceStats() -> PerformanceStats? {
        guard !inferenceTimes.isEmpty else { return nil }
        
        let sortedTimes = inferenceTimes.sorted()
        let count = sortedTimes.count
        
        let averageTime = inferenceTimes.reduce(0, +) / Double(count)
        let minTime = sortedTimes.first!
        let maxTime = sortedTimes.last!
        let medianTime = count % 2 == 0 
            ? (sortedTimes[count/2 - 1] + sortedTimes[count/2]) / 2
            : sortedTimes[count/2]
        
        return PerformanceStats(
            averageTime: averageTime,
            minTime: minTime,
            maxTime: maxTime,
            medianTime: medianTime,
            sampleCount: count
        )
    }
    
    /**
     * Check if performance is within acceptable bounds
     */
    static func isPerformanceAcceptable(targetTime: Double = 0.1) -> Bool {
        guard let stats = getPerformanceStats() else { return true }
        return stats.averageTime <= targetTime
    }
    
    /**
     * Get optimization suggestions based on performance
     */
    static func getOptimizationSuggestions() -> [String] {
        guard let stats = getPerformanceStats() else { return [] }
        
        var suggestions: [String] = []
        
        if stats.averageTime > 0.2 {
            suggestions.append("Consider reducing input image size or model complexity")
        }
        
        if stats.maxTime > stats.averageTime * 2 {
            suggestions.append("High variance in inference times - check for background processing interference")
        }
        
        if stats.averageTime > 0.1 {
            suggestions.append("Enable CoreML execution provider if available")
            suggestions.append("Reduce number of inference threads on older devices")
        }
        
        return suggestions
    }
}

/**
 * Performance statistics for monitoring model efficiency
 */
struct PerformanceStats {
    let averageTime: Double    // Average inference time in seconds
    let minTime: Double       // Fastest inference time
    let maxTime: Double       // Slowest inference time
    let medianTime: Double    // Median inference time
    let sampleCount: Int      // Number of recorded inferences
    
    var formattedSummary: String {
        return """
        Performance Summary (\(sampleCount) samples):
        Average: \(String(format: "%.1f", averageTime * 1000))ms
        Median: \(String(format: "%.1f", medianTime * 1000))ms
        Range: \(String(format: "%.1f", minTime * 1000))-\(String(format: "%.1f", maxTime * 1000))ms
        """
    }
}
