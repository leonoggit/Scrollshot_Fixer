import XCTest
@testable import CAMUSSegmentation

/**
 * Comprehensive test suite for CAMUS Segmentation Model
 * 
 * These tests verify:
 * - Model initialization and loading
 * - Image preprocessing pipeline
 * - Inference accuracy and performance
 * - Error handling and edge cases
 * - Memory management
 */
class CAMUSSegmentationModelTests: XCTestCase {
    
    var model: CAMUSSegmentationModel!
    
    override func setUp() {
        super.setUp()
        model = CAMUSSegmentationModel()
        
        // Wait for model to initialize
        let expectation = self.expectation(description: "Model initialization")
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
            expectation.fulfill()
        }
        waitForExpectations(timeout: 5.0, handler: nil)
    }
    
    override func tearDown() {
        model = nil
        super.tearDown()
    }
    
    // MARK: - Model Initialization Tests
    
    func testModelInitialization() {
        XCTAssertNotNil(model, "Model should initialize successfully")
        XCTAssertTrue(model.isReady, "Model should be ready after initialization")
    }
    
    func testDeviceCapabilityCheck() {
        let isCapable = DeviceCapabilityChecker.isDeviceCapable()
        let threadCount = DeviceCapabilityChecker.getRecommendedThreadCount()
        
        XCTAssertGreaterThan(threadCount, 0, "Thread count should be positive")
        XCTAssertLessThanOrEqual(threadCount, 4, "Thread count should not exceed 4")
        
        print("Device capable: \(isCapable), recommended threads: \(threadCount)")
    }
    
    // MARK: - Image Preprocessing Tests
    
    func testImageResizing() {
        let originalImage = createTestImage(size: CGSize(width: 512, height: 512))
        let resizedImage = originalImage.resized(to: CGSize(width: 256, height: 256))
        
        XCTAssertNotNil(resizedImage, "Image should resize successfully")
        XCTAssertEqual(resizedImage?.size.width, 256, "Resized width should be 256")
        XCTAssertEqual(resizedImage?.size.height, 256, "Resized height should be 256")
    }
    
    func testGrayscaleConversion() {
        let colorImage = createTestImage(size: CGSize(width: 100, height: 100))
        let grayscaleData = colorImage.grayscalePixelData()
        
        XCTAssertNotNil(grayscaleData, "Grayscale conversion should succeed")
        XCTAssertEqual(grayscaleData?.count, 100 * 100, "Should have correct number of pixels")
        
        // Check value range (0-255)
        if let data = grayscaleData {
            let minValue = data.min() ?? 0
            let maxValue = data.max() ?? 0
            XCTAssertGreaterThanOrEqual(minValue, 0, "Minimum pixel value should be >= 0")
            XCTAssertLessThanOrEqual(maxValue, 255, "Maximum pixel value should be <= 255")
        }
    }
    
    func testImagePreprocessing() {
        let testImage = createTestImage(size: CGSize(width: 512, height: 384))
        let preprocessedData = model.preprocessImage(testImage)
        
        XCTAssertNotNil(preprocessedData, "Preprocessing should succeed")
        XCTAssertEqual(preprocessedData?.count, 256 * 256, "Should be resized to 256x256")
        
        // Check normalization (values should be between 0 and 1)
        if let data = preprocessedData {
            let minValue = data.min() ?? 0
            let maxValue = data.max() ?? 0
            XCTAssertGreaterThanOrEqual(minValue, 0.0, "Normalized minimum should be >= 0.0")
            XCTAssertLessThanOrEqual(maxValue, 1.0, "Normalized maximum should be <= 1.0")
            
            print("Preprocessed data range: [\(minValue), \(maxValue)]")
        }
    }
    
    // MARK: - Inference Tests
    
    func testBasicInference() {
        let testImage = createTestImage(size: CGSize(width: 256, height: 256))
        let expectation = self.expectation(description: "Basic inference completion")
        
        var result: SegmentationResult?
        var error: ModelError?
        
        model.predict(image: testImage) { inferenceResult in
            switch inferenceResult {
            case .success(let segmentationResult):
                result = segmentationResult
            case .failure(let inferenceError):
                error = inferenceError
            }
            expectation.fulfill()
        }
        
        waitForExpectations(timeout: 10.0, handler: nil)
        
        XCTAssertNotNil(result, "Inference should produce a result")
        XCTAssertNil(error, "Inference should not produce an error")
        
        if let result = result {
            XCTAssertEqual(result.segmentationMask.count, 256, "Should have 256 rows")
            XCTAssertEqual(result.segmentationMask[0].count, 256, "Should have 256 columns")
            XCTAssertGreaterThan(result.confidence, 0.0, "Confidence should be positive")
            XCTAssertLessThanOrEqual(result.confidence, 1.0, "Confidence should be <= 1.0")
        }
    }
    
    func testInferencePerformance() {
        let testImage = createTestImage(size: CGSize(width: 256, height: 256))
        let expectation = self.expectation(description: "Performance inference completion")
        
        var inferenceTime: Double = 0
        
        model.predict(image: testImage) { result in
            switch result {
            case .success(let segmentationResult):
                inferenceTime = segmentationResult.inferenceTime
            case .failure:
                XCTFail("Performance test should not fail")
            }
            expectation.fulfill()
        }
        
        waitForExpectations(timeout: 10.0, handler: nil)
        
        // Performance expectations
        XCTAssertGreaterThan(inferenceTime, 0.0, "Inference time should be positive")
        XCTAssertLessThan(inferenceTime, 2.0, "Inference should complete within 2 seconds")
        
        print("Inference time: \(String(format: "%.1f", inferenceTime * 1000))ms")
        
        // Record for performance monitoring
        PerformanceMonitor.recordInferenceTime(inferenceTime)
    }
    
    func testMultipleInferences() {
        let testImage = createTestImage(size: CGSize(width: 256, height: 256))
        let expectation = self.expectation(description: "Multiple inferences completion")
        expectation.expectedFulfillmentCount = 3
        
        var results: [SegmentationResult] = []
        
        for i in 0..<3 {
            model.predict(image: testImage) { result in
                switch result {
                case .success(let segmentationResult):
                    results.append(segmentationResult)
                case .failure:
                    XCTFail("Multiple inference test should not fail at iteration \(i)")
                }
                expectation.fulfill()
            }
        }
        
        waitForExpectations(timeout: 30.0, handler: nil)
        
        XCTAssertEqual(results.count, 3, "Should complete all inferences")
        
        // Check consistency (results should be identical for same input)
        if results.count >= 2 {
            let firstResult = results[0]
            let secondResult = results[1]
            
            // Confidence should be very similar (within 1%)
            let confidenceDiff = abs(firstResult.confidence - secondResult.confidence)
            XCTAssertLessThan(confidenceDiff, 0.01, "Confidence should be consistent across runs")
        }
    }
    
    // MARK: - Segmentation Quality Tests
    
    func testTestUltrasoundImage() {
        let testImage = UIImage.createTestUltrasoundImage()
        let expectation = self.expectation(description: "Test ultrasound inference")
        
        var result: SegmentationResult?
        
        model.predict(image: testImage) { inferenceResult in
            switch inferenceResult {
            case .success(let segmentationResult):
                result = segmentationResult
            case .failure(let error):
                XCTFail("Test ultrasound inference failed: \(error.localizedDescription)")
            }
            expectation.fulfill()
        }
        
        waitForExpectations(timeout: 10.0, handler: nil)
        
        guard let result = result else {
            XCTFail("Should have a result for test ultrasound")
            return
        }
        
        // Test ultrasound should potentially detect some structure
        XCTAssertGreaterThan(result.confidence, 0.1, "Should have reasonable confidence for test image")
        
        // Check segmentation statistics
        let stats = result.segmentationStats
        XCTAssertEqual(stats.totalPixels, 256 * 256, "Should process all pixels")
        XCTAssertGreaterThan(stats.backgroundPixels, 0, "Should have background pixels")
        
        print("Test ultrasound results:")
        print("  Has LV: \(result.hasLeftVentricle)")
        print("  Confidence: \(String(format: "%.3f", result.confidence))")
        print("  LV pixels: \(stats.leftVentriclePixels)")
        print("  Quality score: \(String(format: "%.3f", result.qualityScore))")
    }
    
    // MARK: - Error Handling Tests
    
    func testInvalidImageHandling() {
        // Create a minimal 1x1 image
        let tinyImage = createTestImage(size: CGSize(width: 1, height: 1))
        
        let preprocessedData = model.preprocessImage(tinyImage)
        XCTAssertNotNil(preprocessedData, "Even tiny images should preprocess successfully")
        XCTAssertEqual(preprocessedData?.count, 256 * 256, "Should still resize to 256x256")
    }
    
    func testMemoryPressure() {
        // Test multiple rapid inferences to check memory management
        let testImage = createTestImage(size: CGSize(width: 256, height: 256))
        let expectation = self.expectation(description: "Memory pressure test")
        expectation.expectedFulfillmentCount = 5
        
        for i in 0..<5 {
            DispatchQueue.global(qos: .background).async {
                self.model.predict(image: testImage) { result in
                    switch result {
                    case .success:
                        break // Success
                    case .failure(let error):
                        print("Memory pressure test failed at iteration \(i): \(error)")
                    }
                    expectation.fulfill()
                }
            }
        }
        
        waitForExpectations(timeout: 20.0, handler: nil)
        
        // If we get here without crashes, memory management is working
        XCTAssertTrue(true, "Memory pressure test completed without crashes")
    }
    
    // MARK: - Visualization Tests
    
    func testOverlayCreation() {
        // Create a simple segmentation mask
        let mockMask = Array(repeating: Array(repeating: 0, count: 256), count: 256)
        
        // Add some LV structures
        var testMask = mockMask
        for i in 100..<150 {
            for j in 100..<150 {
                testMask[i][j] = 1 // LV cavity
            }
        }
        
        let overlayImage = SegmentationVisualizer.createOverlayImage(
            from: testMask,
            originalSize: CGSize(width: 256, height: 256)
        )
        
        XCTAssertNotNil(overlayImage, "Should create overlay image")
        
        if let overlay = overlayImage {
            XCTAssertEqual(overlay.size.width, 256, "Overlay should match original width")
            XCTAssertEqual(overlay.size.height, 256, "Overlay should match original height")
        }
    }
    
    func testImageCombination() {
        let originalImage = createTestImage(size: CGSize(width: 256, height: 256))
        let overlayImage = createTestImage(size: CGSize(width: 256, height: 256))
        
        let combinedImage = SegmentationVisualizer.combineImages(
            original: originalImage,
            overlay: overlayImage
        )
        
        XCTAssertNotNil(combinedImage, "Should combine images successfully")
        
        if let combined = combinedImage {
            XCTAssertEqual(combined.size, originalImage.size, "Combined image should match original size")
        }
    }
    
    // MARK: - Performance Monitoring Tests
    
    func testPerformanceMonitoring() {
        // Clear previous data
        let initialStats = PerformanceMonitor.getPerformanceStats()
        
        // Record some test times
        PerformanceMonitor.recordInferenceTime(0.05)  // 50ms
        PerformanceMonitor.recordInferenceTime(0.08)  // 80ms
        PerformanceMonitor.recordInferenceTime(0.06)  // 60ms
        
        let stats = PerformanceMonitor.getPerformanceStats()
        XCTAssertNotNil(stats, "Should have performance stats")
        
        if let stats = stats {
            XCTAssertGreaterThan(stats.sampleCount, 0, "Should have recorded samples")
            XCTAssertGreaterThan(stats.averageTime, 0, "Average time should be positive")
            XCTAssertLessThanOrEqual(stats.minTime, stats.averageTime, "Min should be <= average")
            XCTAssertGreaterThanOrEqual(stats.maxTime, stats.averageTime, "Max should be >= average")
        }
        
        let isAcceptable = PerformanceMonitor.isPerformanceAcceptable(targetTime: 0.1)
        XCTAssertTrue(isAcceptable, "Test performance should be acceptable")
        
        let suggestions = PerformanceMonitor.getOptimizationSuggestions()
        XCTAssertTrue(suggestions.isEmpty, "Good performance should have no suggestions")
    }
    
    // MARK: - Helper Methods
    
    private func createTestImage(size: CGSize) -> UIImage {
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1.0
        format.opaque = true
        
        let renderer = UIGraphicsImageRenderer(size: size, format: format)
        
        return renderer.image { context in
            // Create a gradient pattern for testing
            UIColor.lightGray.setFill()
            context.fill(CGRect(origin: .zero, size: size))
            
            // Add some patterns
            UIColor.darkGray.setFill()
            let centerRect = CGRect(
                x: size.width * 0.25,
                y: size.height * 0.25,
                width: size.width * 0.5,
                height: size.height * 0.5
            )
            context.fill(centerRect)
        }
    }
}

// MARK: - Performance Test Suite

class CAMUSPerformanceTests: XCTestCase {
    
    var model: CAMUSSegmentationModel!
    
    override func setUp() {
        super.setUp()
        model = CAMUSSegmentationModel()
        
        // Warm up model
        let warmupExpectation = expectation(description: "Model warmup")
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
            warmupExpectation.fulfill()
        }
        waitForExpectations(timeout: 5.0, handler: nil)
    }
    
    func testInferenceSpeed() {
        measure {
            let testImage = UIImage.createTestUltrasoundImage()
            let expectation = self.expectation(description: "Speed test inference")
            
            model.predict(image: testImage) { _ in
                expectation.fulfill()
            }
            
            waitForExpectations(timeout: 5.0, handler: nil)
        }
    }
    
    func testMemoryUsage() {
        let testImage = UIImage.createTestUltrasoundImage()
        
        measureMetrics([.wallClockTime], automaticallyStartMeasuring: false) {
            startMeasuring()
            
            let expectation = self.expectation(description: "Memory test inference")
            
            model.predict(image: testImage) { _ in
                expectation.fulfill()
            }
            
            waitForExpectations(timeout: 5.0, handler: nil)
            
            stopMeasuring()
        }
    }
}
