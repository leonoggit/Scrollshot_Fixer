import Foundation
import onnxruntime_objc

/**
 * CAMUS Left Ventricle Segmentation Model
 * 
 * This class provides a Swift wrapper for our converted ONNX segmentation model.
 * The model performs left ventricle segmentation on cardiac ultrasound images.
 * 
 * Model specifications:
 * - Input: [1, 1, 256, 256] - Single channel (grayscale) 256x256 image
 * - Output: [1, 3, 256, 256] - 3 classes (background, LV cavity, LV wall)
 * - Model size: 56MB
 * - Expected inference time: 50-100ms on iPhone 14/15
 */
class CAMUSSegmentationModel {
    
    // MARK: - Properties
    
    private var session: ORTSession?
    private let modelName = "camus_segmentation_real_weights"
    
    // Model specifications (from our successful conversion)
    private let inputShape = [1, 1, 256, 256]  // [batch, channel, height, width]
    private let outputShape = [1, 3, 256, 256] // [batch, classes, height, width]
    
    private let expectedInputName = "input"     // Adjust based on actual model
    private let expectedOutputName = "output"   // Adjust based on actual model

    // Normalization constants from training dataset
    private let intensityMean: Float = 76.2786
    private let intensityStd: Float = 47.6041
    
    // MARK: - Initialization
    
    init() {
        setupModel()
    }
    
    deinit {
        session = nil
        print("🧹 CAMUS segmentation model deallocated")
    }
    
    // MARK: - Model Setup
    
    private func setupModel() {
        guard let modelPath = Bundle.main.path(forResource: modelName, ofType: "onnx") else {
            print("❌ Error: Could not find \(modelName).onnx in bundle")
            print("   Make sure the model file is added to your Xcode project target")
            return
        }
        
        do {
            let env = try ORTEnv(loggingLevel: ORTLoggingLevel.warning)
            let options = try ORTSessionOptions()
            
            // Performance optimizations
            try options.setLogSeverityLevel(ORTLoggingLevel.warning)
            try options.setIntraOpNumThreads(DeviceCapabilityChecker.getRecommendedThreadCount())
            try options.setInterOpNumThreads(1)
            try options.setGraphOptimizationLevel(.all)
            
            // Memory optimizations
            try options.addConfigEntry("session.memory.enable_memory_pattern", value: "1")
            try options.addConfigEntry("session.memory.enable_memory_arena_shrinkage", value: "1")
            
            // iOS-specific optimizations
            #if !targetEnvironment(simulator)
            // Try to use CoreML execution provider on device
            do {
                try options.appendExecutionProvider("CoreML", options: [:])
                print("✅ CoreML execution provider enabled")
            } catch {
                print("⚠️ CoreML execution provider not available, using CPU")
            }
            #endif
            
            session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: options)
            print("✅ ONNX segmentation model loaded successfully")
            
            // Log model information
            logModelInfo()
            
        } catch {
            print("❌ Error creating ONNX session: \(error)")
            print("   Check that ONNX Runtime is properly installed via CocoaPods or SPM")
        }
    }
    
    private func logModelInfo() {
        guard let session = session else { return }
        
        do {
            let inputNames = try session.inputNames()
            let outputNames = try session.outputNames()
            
            print("📋 CAMUS Segmentation Model Info:")
            print("   Model file: \(modelName).onnx")
            print("   Input names: \(inputNames)")
            print("   Output names: \(outputNames)")
            print("   Expected input shape: \(inputShape)")
            print("   Expected output shape: \(outputShape)")
            print("   Target classes: 0=Background, 1=LV Cavity, 2=LV Wall")
            
        } catch {
            print("❌ Error getting model info: \(error)")
        }
    }
    
    // MARK: - Public Interface
    
    /**
     * Perform left ventricle segmentation on a cardiac ultrasound image
     * 
     * - Parameter image: Input ultrasound image (will be resized to 256x256)
     * - Parameter completion: Completion handler with segmentation result
     */
    func predict(image: UIImage, completion: @escaping (Result<SegmentationResult, ModelError>) -> Void) {
        // Perform inference on background queue to avoid blocking UI
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { 
                DispatchQueue.main.async {
                    completion(.failure(.sessionNotInitialized))
                }
                return 
            }
            
            do {
                let result = try self.performInference(image: image)
                DispatchQueue.main.async {
                    completion(.success(result))
                }
            } catch let error as ModelError {
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
            } catch {
                DispatchQueue.main.async {
                    completion(.failure(.unknownError(error.localizedDescription)))
                }
            }
        }
    }
    
    /**
     * Warm up the model with a dummy inference to reduce first-prediction latency
     */
    func warmUp() {
        let dummyImage = createDummyImage()
        predict(image: dummyImage) { result in
            switch result {
            case .success:
                print("🔥 CAMUS segmentation model warmed up successfully")
            case .failure(let error):
                print("⚠️ Model warm-up failed: \(error.localizedDescription)")
            }
        }
    }
    
    /**
     * Check if the model is ready for inference
     */
    var isReady: Bool {
        return session != nil
    }
    
    // MARK: - Private Methods
    
    private func performInference(image: UIImage) throws -> SegmentationResult {
        guard let session = session else {
            throw ModelError.sessionNotInitialized
        }
        
        // Step 1: Preprocess image
        guard let inputData = preprocessImage(image) else {
            throw ModelError.preprocessingFailed
        }
        
        // Step 2: Create input tensor
        let inputNames = try session.inputNames()
        guard !inputNames.isEmpty else {
            throw ModelError.invalidModel("No input names found")
        }
        
        let inputName = inputNames[0] // Use first input
        let shape: [NSNumber] = inputShape.map { NSNumber(value: $0) }
        
        let inputTensor = try ORTValue(
            tensorData: NSMutableData(data: Data(bytes: inputData, count: inputData.count * MemoryLayout<Float>.size)),
            elementType: ORTTensorElementDataType.float,
            shape: shape
        )
        
        // Step 3: Run inference
        let startTime = CFAbsoluteTimeGetCurrent()
        let outputs = try session.run(
            withInputs: [inputName: inputTensor],
            outputNames: try session.outputNames(),
            runOptions: nil
        )
        let inferenceTime = CFAbsoluteTimeGetCurrent() - startTime
        
        // Step 4: Process output
        let outputNames = try session.outputNames()
        guard !outputNames.isEmpty else {
            throw ModelError.outputProcessingFailed("No output names found")
        }
        
        guard let outputTensor = outputs[outputNames[0]] else {
            throw ModelError.outputProcessingFailed("Output tensor not found")
        }
        
        let result = try processOutput(outputTensor, inferenceTime: inferenceTime)
        return result
    }
    
    private func createDummyImage() -> UIImage {
        let size = CGSize(width: 256, height: 256)
        UIGraphicsBeginImageContext(size)
        UIColor.black.setFill()
        UIRectFill(CGRect(origin: .zero, size: size))
        let image = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()
        return image
    }
}

// MARK: - Image Preprocessing Extension

extension CAMUSSegmentationModel {
    
    /**
     * Preprocess image for model input
     * - Resize to 256x256
     * - Convert to grayscale
     * - Normalize to [0, 1] range
     */
    func preprocessImage(_ image: UIImage) -> [Float]? {
        // Step 1: Resize to model input size (256x256)
        guard let resizedImage = image.resized(to: CGSize(width: 256, height: 256)) else {
            print("❌ Error resizing image to 256x256")
            return nil
        }
        
        // Step 2: Convert to grayscale and extract pixel data
        guard let pixelData = resizedImage.grayscalePixelData() else {
            print("❌ Error extracting grayscale pixel data")
            return nil
        }
        
        // Step 3: Z-score normalization using the training dataset statistics
        let normalizedData = pixelData.map { (Float($0) - intensityMean) / intensityStd }
        
        // Verify data size
        let expectedSize = 256 * 256
        guard normalizedData.count == expectedSize else {
            print("❌ Unexpected data size: \(normalizedData.count), expected: \(expectedSize)")
            return nil
        }
        
        print("✅ Image preprocessed: \(normalizedData.count) pixels, range: [\(normalizedData.min() ?? 0), \(normalizedData.max() ?? 1)]")
        
        return normalizedData
    }
}

// MARK: - Output Processing Extension

extension CAMUSSegmentationModel {
    
    private func processOutput(_ outputTensor: ORTValue, inferenceTime: Double) throws -> SegmentationResult {
        // Get tensor data as raw bytes
        let tensorData = try outputTensor.tensorData() as Data
        let floatArray = tensorData.withUnsafeBytes { bytes in
            return Array(bytes.bindMemory(to: Float.self))
        }
        
        // Verify output size
        let expectedOutputSize = outputShape.reduce(1, *)
        guard floatArray.count == expectedOutputSize else {
            throw ModelError.outputProcessingFailed("Unexpected output size: \(floatArray.count), expected: \(expectedOutputSize)")
        }
        
        // Output shape: [1, 3, 256, 256] - 3 classes (background, LV cavity, LV wall)
        let batchSize = 1
        let numClasses = 3
        let height = 256
        let width = 256
        
        // Apply softmax and get segmentation mask
        let segmentationMask = applySoftmaxAndGetMask(
            predictions: floatArray,
            numClasses: numClasses,
            height: height,
            width: width
        )
        
        // Calculate prediction confidence
        let confidence = calculateAverageConfidence(
            predictions: floatArray,
            numClasses: numClasses,
            height: height,
            width: width
        )
        
        // Calculate segmentation statistics
        let stats = calculateSegmentationStats(segmentationMask)
        
        print("✅ Segmentation complete: \(String(format: "%.1f", inferenceTime * 1000))ms, confidence: \(String(format: "%.3f", confidence))")
        print("   LV pixels: \(stats.leftVentriclePixels), cavity: \(stats.cavityPixels), wall: \(stats.wallPixels)")
        
        // Create result
        return SegmentationResult(
            segmentationMask: segmentationMask,
            confidence: confidence,
            inferenceTime: inferenceTime,
            imageSize: CGSize(width: width, height: height),
            segmentationStats: stats
        )
    }
    
    private func applySoftmaxAndGetMask(predictions: [Float], numClasses: Int, height: Int, width: Int) -> [[Int]] {
        var segmentationMask = Array(repeating: Array(repeating: 0, count: width), count: height)
        
        for h in 0..<height {
            for w in 0..<width {
                var maxLogit: Float = -Float.infinity
                var bestClass = 0
                
                // Find class with highest logit (before softmax)
                for c in 0..<numClasses {
                    let index = c * height * width + h * width + w
                    let logit = predictions[index]
                    
                    if logit > maxLogit {
                        maxLogit = logit
                        bestClass = c
                    }
                }
                
                segmentationMask[h][w] = bestClass
            }
        }
        
        return segmentationMask
    }
    
    private func calculateAverageConfidence(predictions: [Float], numClasses: Int, height: Int, width: Int) -> Float {
        var totalConfidence: Float = 0.0
        let numPixels = height * width
        
        for h in 0..<height {
            for w in 0..<width {
                // Calculate softmax for this pixel
                var maxLogit: Float = -Float.infinity
                for c in 0..<numClasses {
                    let index = c * height * width + h * width + w
                    maxLogit = max(maxLogit, predictions[index])
                }
                
                // Calculate softmax probabilities
                var probSum: Float = 0.0
                var maxProb: Float = 0.0
                
                for c in 0..<numClasses {
                    let index = c * height * width + h * width + w
                    let prob = exp(predictions[index] - maxLogit) // Numerical stability
                    probSum += prob
                    maxProb = max(maxProb, prob)
                }
                
                // Normalized max probability
                let confidence = maxProb / probSum
                totalConfidence += confidence
            }
        }
        
        return totalConfidence / Float(numPixels)
    }
    
    private func calculateSegmentationStats(_ mask: [[Int]]) -> SegmentationStats {
        var backgroundPixels = 0
        var cavityPixels = 0
        var wallPixels = 0
        
        for row in mask {
            for pixel in row {
                switch pixel {
                case 0: backgroundPixels += 1
                case 1: cavityPixels += 1
                case 2: wallPixels += 1
                default: break
                }
            }
        }
        
        let leftVentriclePixels = cavityPixels + wallPixels
        let totalPixels = backgroundPixels + leftVentriclePixels
        
        return SegmentationStats(
            backgroundPixels: backgroundPixels,
            cavityPixels: cavityPixels,
            wallPixels: wallPixels,
            leftVentriclePixels: leftVentriclePixels,
            totalPixels: totalPixels
        )
    }
}

// MARK: - Device Capability Checker

class DeviceCapabilityChecker {
    
    static func isDeviceCapable() -> Bool {
        let device = UIDevice.current
        
        // Check available memory (minimum 3GB for smooth operation)
        let minimumRAM: UInt64 = 3 * 1024 * 1024 * 1024 // 3GB in bytes
        let currentRAM = ProcessInfo.processInfo.physicalMemory
        
        let hasEnoughRAM = currentRAM >= minimumRAM
        
        // Check iOS version (minimum iOS 13 for ONNX Runtime)
        let minimumIOSVersion = "13.0"
        let currentIOSVersion = device.systemVersion
        let hasCompatibleOS = currentIOSVersion.compare(minimumIOSVersion, options: .numeric) != .orderedAscending
        
        return hasEnoughRAM && hasCompatibleOS
    }
    
    static func getRecommendedThreadCount() -> Int {
        let processorCount = ProcessInfo.processInfo.processorCount
        
        // Use half of available cores, but cap at 4 for stability
        let recommendedCount = max(1, min(processorCount / 2, 4))
        
        print("📱 Device has \(processorCount) cores, using \(recommendedCount) threads")
        return recommendedCount
    }
    
    static func getDeviceInfo() -> String {
        let device = UIDevice.current
        let ram = ProcessInfo.processInfo.physicalMemory / (1024 * 1024 * 1024)
        let cores = ProcessInfo.processInfo.processorCount
        
        return """
        Device: \(device.model)
        iOS: \(device.systemVersion)
        RAM: \(ram)GB
        Cores: \(cores)
        Capable: \(isDeviceCapable())
        """
    }
}
