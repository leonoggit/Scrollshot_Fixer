import SwiftUI
import AVFoundation
import Combine

class UltrasoundSegmentationProcessor: ObservableObject {
    @Published var processingProgress: Double = 0.0
    @Published var isProcessingComplete: Bool = false
    @Published var segmentationResults: [FrameSegmentationResult] = []
    @Published var currentStatus: String = "Inicializando..."
    @Published var errorMessage: String?
    
    private let segmentationModel: CAMUSSegmentationModel
    private var processingTask: Task<Void, Never>?
    
    init() {
        self.segmentationModel = CAMUSSegmentationModel()
    }
    
    func processVideo(url: URL) {
        reset()
        
        processingTask = Task {
            await performVideoSegmentation(url: url)
        }
    }
    
    func reset() {
        processingTask?.cancel()
        
        DispatchQueue.main.async {
            self.processingProgress = 0.0
            self.isProcessingComplete = false
            self.segmentationResults = []
            self.currentStatus = "Inicializando..."
            self.errorMessage = nil
        }
    }
    
    @MainActor
    private func updateStatus(_ status: String) {
        self.currentStatus = status
    }
    
    @MainActor
    private func updateProgress(_ progress: Double) {
        self.processingProgress = progress
    }
    
    @MainActor
    private func setError(_ error: String) {
        self.errorMessage = error
    }
    
    @MainActor
    private func setComplete() {
        self.isProcessingComplete = true
        self.currentStatus = "Processamento concluído!"
    }
    
    private func performVideoSegmentation(url: URL) async {
        await updateStatus("Carregando vídeo...")
        
        // Start accessing security-scoped resource
        let isAccessingSecurityScopedResource = url.startAccessingSecurityScopedResource()
        defer {
            if isAccessingSecurityScopedResource {
                url.stopAccessingSecurityScopedResource()
            }
        }
        
        let asset = AVAsset(url: url)
        
        do {
            // Get video properties
            let duration = try await asset.load(.duration)
            let tracks = try await asset.load(.tracks)
            
            // Get video properties
            guard tracks.first(where: { $0.mediaType == .video }) != nil else {
                await setError("Nenhuma trilha de vídeo encontrada no arquivo selecionado")
                return
            }
            
            await updateStatus("Extraindo quadros...")
            
            // Create image generator
            let imageGenerator = AVAssetImageGenerator(asset: asset)
            imageGenerator.appliesPreferredTrackTransform = true
            imageGenerator.requestedTimeToleranceAfter = .zero
            imageGenerator.requestedTimeToleranceBefore = .zero
            
            // Calculate frame extraction parameters
            let frameRate: Double = 5.0 // Extract 5 frames per second
            let durationSeconds = duration.seconds
            let totalFrames = Int(durationSeconds * frameRate)
            
            await updateStatus("Processando \(totalFrames) quadros...")
            
            // Create a local array for collecting results, avoiding capture in concurrent context
            var localResults: [FrameSegmentationResult] = []
            
            // Process frames
            for frameIndex in 0..<totalFrames {
                if Task.isCancelled {
                    break
                }
                
                let timeValue = CMTime(seconds: Double(frameIndex) / frameRate, preferredTimescale: 600)
                
                do {
                    // Extract frame
                    let cgImage = try imageGenerator.copyCGImage(at: timeValue, actualTime: nil)
                    let uiImage = UIImage(cgImage: cgImage)
                    
                    await updateStatus("Segmentando quadro \(frameIndex + 1)/\(totalFrames)...")
                    
                    // Perform segmentation
                    let segmentationResult = try segmentationModel.performSegmentation(on: uiImage)
                    
                    // Create visualization
                    let visualizedImage = createSegmentationVisualization(
                        original: uiImage,
                        segmentation: segmentationResult
                    )
                    
                    let frameResult = FrameSegmentationResult(
                        frameIndex: frameIndex,
                        timestamp: timeValue.seconds,
                        originalImage: uiImage,
                        segmentationResult: segmentationResult,
                        visualizedImage: visualizedImage
                    )
                    
                    // Add to local results array
                    localResults.append(frameResult)
                    
                    // Update progress
                    let progress = Double(frameIndex + 1) / Double(totalFrames)
                    await updateProgress(progress)
                    
                } catch {
                    print("Falha ao processar quadro \(frameIndex): \(error)")
                    // Continue with next frame instead of failing completely
                }
                
                // Small delay to prevent overwhelming the system
                try? await Task.sleep(nanoseconds: 10_000_000) // 10ms
            }
            
            // Update results on main actor to avoid concurrency issues
            await updateResults(with: localResults)
            
            await setComplete()
            
        } catch {
            await setError("Falha no processamento do vídeo: \(error.localizedDescription)")
        }
    }
    
    @MainActor
    private func updateResults(with results: [FrameSegmentationResult]) {
        self.segmentationResults = results
    }
    
    private func createSegmentationVisualization(
        original: UIImage,
        segmentation: SegmentationResult
    ) -> UIImage {
        // Create overlay image from segmentation mask
        guard let overlayImage = SegmentationVisualizer.createOverlayImage(
            from: segmentation.segmentationMask,
            originalSize: original.size,
            alpha: 0.5
        ) else {
            return original
        }
        
        // Combine original with overlay
        return SegmentationVisualizer.combineImages(
            original: original,
            overlay: overlayImage,
            overlayAlpha: 0.5
        ) ?? original
    }
}

struct FrameSegmentationResult: Identifiable {
    let id = UUID()
    let frameIndex: Int
    let timestamp: Double
    let originalImage: UIImage
    let segmentationResult: SegmentationResult
    let visualizedImage: UIImage
    
    var formattedTimestamp: String {
        let minutes = Int(timestamp) / 60
        let seconds = Int(timestamp) % 60
        let milliseconds = Int((timestamp.truncatingRemainder(dividingBy: 1)) * 1000)
        
        if minutes > 0 {
            return String(format: "%d:%02d.%03d", minutes, seconds, milliseconds)
        } else {
            return String(format: "%d.%03d", seconds, milliseconds)
        }
    }
}
