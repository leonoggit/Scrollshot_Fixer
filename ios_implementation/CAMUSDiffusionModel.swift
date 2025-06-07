import Foundation
import onnxruntime_objc
import UIKit

/**
 * CAMUS Diffusion Enhancement Model
 *
 * This Swift class provides an interface to the converted diffusion ONNX model.
 * The model takes a 256x256 RGB image and outputs an enhanced image with 6
 * channels (learned sigma variant). It mirrors the segmentation model wrapper
 * so it can be easily integrated in the demo app.
 */
class CAMUSDiffusionModel {
    private var session: ORTSession?
    private let modelName = "camus_diffusion_model"
    private let inputShape = [1, 3, 256, 256]
    private let outputShape = [1, 6, 256, 256]

    init() {
        setupModel()
    }

    deinit {
        session = nil
        print("🧹 CAMUS diffusion model deallocated")
    }

    private func setupModel() {
        guard let modelPath = Bundle.main.path(forResource: modelName, ofType: "onnx") else {
            print("❌ Diffusion model not found in bundle")
            return
        }

        do {
            let env = try ORTEnv(loggingLevel: ORTLoggingLevel.warning)
            let options = try ORTSessionOptions()
            try options.setGraphOptimizationLevel(.all)
            try options.setIntraOpNumThreads(DeviceCapabilityChecker.getRecommendedThreadCount())
            #if !targetEnvironment(simulator)
            do {
                try options.appendExecutionProvider("CoreML", options: [:])
                print("✅ CoreML EP enabled for diffusion model")
            } catch {
                print("⚠️ CoreML EP unavailable, using CPU")
            }
            #endif

            session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: options)
            print("✅ Diffusion ONNX model loaded")
        } catch {
            print("❌ Failed to load diffusion model: \(error)")
        }
    }

    /**
     * Enhance an image using the diffusion model.
     * - Parameter image: Input UIImage (will be resized to 256x256 and normalized)
     * - Returns: Enhanced UIImage or nil if inference fails
     */
    func enhance(image: UIImage) -> UIImage? {
        guard let session = session else { return nil }
        guard let resized = image.resized(to: CGSize(width: 256, height: 256)) else { return nil }
        guard let rgbData = resized.rgbData() else { return nil }

        do {
            let tensor = try ORTValue(tensorData: Data(buffer: rgbData.withUnsafeBytes { $0 }),
                                     elementType: ORTTensorElementDataType.float,
                                     shape: inputShape as [NSNumber])
            let input = ["input": tensor]
            let output = try session.run(withInputs: input, outputNames: ["output"]) as? [ORTValue]
            guard let first = output?.first,
                  let floatArray = first.tensorData()?.withUnsafeBytes({ Data($0) }) else {
                return nil
            }
            // Simple post-processing: convert first three channels back to image
            let count = 3 * 256 * 256
            let buffer = floatArray.withUnsafeBytes { ptr in
                Array(ptr.bindMemory(to: Float.self))
            }
            let clipped = buffer.prefix(count)
            return UIImage.fromRGBFloats(Array(clipped), width: 256, height: 256)
        } catch {
            print("❌ Diffusion inference failed: \(error)")
            return nil
        }
    }
}

// MARK: - UIImage helpers
extension UIImage {
    func rgbData() -> [Float]? {
        guard let cgImage = self.cgImage else { return nil }
        let width = cgImage.width
        let height = cgImage.height
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var pixelData = [Float](repeating: 0, count: width * height * 3)
        let context = CGContext(data: &pixelData,
                                width: width,
                                height: height,
                                bitsPerComponent: 8,
                                bytesPerRow: width * 3 * 4,
                                space: colorSpace,
                                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue)
        context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        // Normalize to [0,1]
        var floats = [Float](repeating: 0, count: width * height * 3)
        var index = 0
        for i in stride(from: 0, to: pixelData.count, by: 4) {
            floats[index] = pixelData[i] / 255.0
            floats[index + 1] = pixelData[i + 1] / 255.0
            floats[index + 2] = pixelData[i + 2] / 255.0
            index += 3
        }
        return floats
    }

    static func fromRGBFloats(_ data: [Float], width: Int, height: Int) -> UIImage? {
        var bytes = data.map { UInt8(max(0, min(255, Int($0 * 255.0)))) }
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let provider = CGDataProvider(data: NSData(bytes: &bytes, length: bytes.count)) else { return nil }
        if let cgImage = CGImage(width: width,
                                 height: height,
                                 bitsPerComponent: 8,
                                 bitsPerPixel: 24,
                                 bytesPerRow: width * 3,
                                 space: colorSpace,
                                 bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                                 provider: provider,
                                 decode: nil,
                                 shouldInterpolate: true,
                                 intent: .defaultIntent) {
            return UIImage(cgImage: cgImage)
        }
        return nil
    }
}

