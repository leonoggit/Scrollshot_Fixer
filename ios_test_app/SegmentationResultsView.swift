import SwiftUI
import AVFoundation

struct SegmentationResultsView: View {
    let videoURL: URL
    let segmentationResults: [FrameSegmentationResult]
    let onDismiss: () -> Void
    
    @State private var currentFrameIndex: Int = 0
    @State private var isPlaying: Bool = false
    @State private var playbackTimer: Timer?
    @State private var showingOverlay: Bool = true
    @State private var selectedViewMode: ViewMode = .sideBySide
    @State private var playbackSpeed: Double = 1.0
    
    enum ViewMode: String, CaseIterable {
        case original = "Original"
        case segmented = "Segmented"
        case sideBySide = "Side by Side"
        case overlay = "Overlay"
    }
    
    var currentFrame: FrameSegmentationResult? {
        guard currentFrameIndex >= 0 && currentFrameIndex < segmentationResults.count else {
            return nil
        }
        return segmentationResults[currentFrameIndex]
    }
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Video display area
                ZStack {
                    Color.black
                        .aspectRatio(16/9, contentMode: .fit)
                    
                    if let frame = currentFrame {
                        switch selectedViewMode {
                        case .original:
                            Image(uiImage: frame.originalImage)
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                        
                        case .segmented:
                            Image(uiImage: frame.visualizedImage)
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                        
                        case .sideBySide:
                            HStack(spacing: 2) {
                                Image(uiImage: frame.originalImage)
                                    .resizable()
                                    .aspectRatio(contentMode: .fit)
                                    .frame(maxWidth: .infinity)
                                
                                Rectangle()
                                    .fill(Color.white)
                                    .frame(width: 2)
                                
                                Image(uiImage: frame.visualizedImage)
                                    .resizable()
                                    .aspectRatio(contentMode: .fit)
                                    .frame(maxWidth: .infinity)
                            }
                        
                        case .overlay:
                            Image(uiImage: frame.visualizedImage)
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                        }
                    } else {
                        VStack {
                            Image(systemName: "exclamationmark.triangle")
                                .font(.system(size: 40))
                                .foregroundColor(.yellow)
                            Text("No frame data available")
                                .foregroundColor(.white)
                        }
                    }
                    
                    // Overlay controls
                    VStack {
                        HStack {
                            Spacer()
                            
                            // View mode selector
                            Menu {
                                ForEach(ViewMode.allCases, id: \.self) { mode in
                                    Button(mode.rawValue) {
                                        selectedViewMode = mode
                                    }
                                }
                            } label: {
                                HStack {
                                    Text(selectedViewMode.rawValue)
                                    Image(systemName: "chevron.down")
                                }
                                .padding(.horizontal, 12)
                                .padding(.vertical, 6)
                                .background(Color.black.opacity(0.7))
                                .foregroundColor(.white)
                                .cornerRadius(8)
                            }
                            .padding()
                        }
                        
                        Spacer()
                        
                        // Frame info overlay
                        if let frame = currentFrame {
                            VStack(alignment: .leading, spacing: 4) {
                                HStack {
                                    Text("Frame \(frame.frameIndex + 1)/\(segmentationResults.count)")
                                        .font(.caption)
                                        .fontWeight(.medium)
                                    
                                    Spacer()
                                    
                                    Text(frame.formattedTimestamp)
                                        .font(.caption)
                                        .fontWeight(.medium)
                                }
                                
                                HStack {
                                    Label("LV: \(String(format: "%.1f", frame.segmentationResult.leftVentriclePercentage))%", 
                                          systemImage: "heart.fill")
                                        .font(.caption2)
                                    
                                    Spacer()
                                    
                                    Label("Confidence: \(String(format: "%.0f", frame.segmentationResult.confidence * 100))%", 
                                          systemImage: "checkmark.circle")
                                        .font(.caption2)
                                }
                            }
                            .padding(.horizontal, 12)
                            .padding(.vertical, 8)
                            .background(Color.black.opacity(0.7))
                            .foregroundColor(.white)
                            .cornerRadius(8)
                            .padding(.horizontal)
                            .padding(.bottom, 20)
                        }
                    }
                }
                
                // Playback controls
                VStack(spacing: 20) {
                    // Timeline scrubber
                    VStack(spacing: 8) {
                        Slider(
                            value: Binding(
                                get: { Double(currentFrameIndex) },
                                set: { newValue in
                                    currentFrameIndex = Int(newValue)
                                }
                            ),
                            in: 0...Double(max(0, segmentationResults.count - 1)),
                            step: 1
                        )
                        .disabled(isPlaying)
                        
                        HStack {
                            Text("Frame \(currentFrameIndex + 1)")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            
                            Spacer()
                            
                            if let frame = currentFrame {
                                Text(frame.formattedTimestamp)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                    .padding(.horizontal)
                    
                    // Control buttons
                    HStack(spacing: 30) {
                        Button(action: {
                            currentFrameIndex = max(0, currentFrameIndex - 1)
                        }) {
                            Image(systemName: "backward.frame")
                                .font(.title2)
                        }
                        .disabled(currentFrameIndex <= 0 || isPlaying)
                        
                        Button(action: {
                            togglePlayback()
                        }) {
                            Image(systemName: isPlaying ? "pause.circle.fill" : "play.circle.fill")
                                .font(.title)
                        }
                        .disabled(segmentationResults.isEmpty)
                        
                        Button(action: {
                            currentFrameIndex = min(segmentationResults.count - 1, currentFrameIndex + 1)
                        }) {
                            Image(systemName: "forward.frame")
                                .font(.title2)
                        }
                        .disabled(currentFrameIndex >= segmentationResults.count - 1 || isPlaying)
                    }
                    
                    // Playback speed control
                    HStack {
                        Text("Speed:")
                            .font(.caption)
                        
                        Picker("Speed", selection: $playbackSpeed) {
                            Text("0.5x").tag(0.5)
                            Text("1x").tag(1.0)
                            Text("2x").tag(2.0)
                            Text("4x").tag(4.0)
                        }
                        .pickerStyle(.segmented)
                    }
                    .padding(.horizontal)
                }
                .padding(.vertical)
                .background(Color(.systemGroupedBackground))
                
                // Analysis summary
                if !segmentationResults.isEmpty {
                    AnalysisSummaryView(results: segmentationResults)
                        .padding()
                        .background(Color(.systemGroupedBackground))
                }
            }
            .navigationTitle("Segmentation Results")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Done") {
                        stopPlayback()
                        onDismiss()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    ShareLink(
                        item: generateSummaryText(),
                        subject: Text("CAMUS Segmentation Analysis")
                    ) {
                        Image(systemName: "square.and.arrow.up")
                    }
                }
            }
        }
        .onDisappear {
            stopPlayback()
        }
    }
    
    private func togglePlayback() {
        if isPlaying {
            stopPlayback()
        } else {
            startPlayback()
        }
    }
    
    private func startPlayback() {
        guard !segmentationResults.isEmpty else { return }
        
        isPlaying = true
        
        let frameInterval = 1.0 / (5.0 * playbackSpeed) // 5 FPS base rate
        
        playbackTimer = Timer.scheduledTimer(withTimeInterval: frameInterval, repeats: true) { _ in
            if currentFrameIndex >= segmentationResults.count - 1 {
                // Reached end, stop or loop
                currentFrameIndex = 0
            } else {
                currentFrameIndex += 1
            }
        }
    }
    
    private func stopPlayback() {
        isPlaying = false
        playbackTimer?.invalidate()
        playbackTimer = nil
    }
    
    private func generateSummaryText() -> String {
        let avgConfidence = segmentationResults.map { $0.segmentationResult.confidence }.reduce(0, +) / Double(segmentationResults.count)
        let avgLVPercentage = segmentationResults.map { $0.segmentationResult.leftVentriclePercentage }.reduce(0, +) / Double(segmentationResults.count)
        
        return """
        CAMUS Cardiac Segmentation Analysis
        
        Video: \(videoURL.lastPathComponent)
        Frames Analyzed: \(segmentationResults.count)
        Average Confidence: \(String(format: "%.1f", avgConfidence * 100))%
        Average LV Coverage: \(String(format: "%.1f", avgLVPercentage))%
        
        Generated by CAMUS Segmentation App
        """
    }
}

struct AnalysisSummaryView: View {
    let results: [FrameSegmentationResult]
    
    private var averageConfidence: Float {
        let total = results.map { $0.segmentationResult.confidence }.reduce(0, +)
        return total / Float(results.count)
    }
    
    private var averageLVPercentage: Float {
        let total = results.map { $0.segmentationResult.leftVentriclePercentage }.reduce(0, +)
        return total / Float(results.count)
    }
    
    private var qualityAssessment: String {
        if averageConfidence > 0.8 && averageLVPercentage > 15 {
            return "Excellent"
        } else if averageConfidence > 0.6 && averageLVPercentage > 10 {
            return "Good"
        } else if averageConfidence > 0.4 && averageLVPercentage > 5 {
            return "Fair"
        } else {
            return "Poor"
        }
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Analysis Summary")
                .font(.headline)
            
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Frames Analyzed")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("\(results.count)")
                        .font(.title2)
                        .fontWeight(.semibold)
                }
                
                Spacer()
                
                VStack(alignment: .trailing, spacing: 4) {
                    Text("Avg. Confidence")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("\(String(format: "%.0f", averageConfidence * 100))%")
                        .font(.title2)
                        .fontWeight(.semibold)
                }
                
                Spacer()
                
                VStack(alignment: .trailing, spacing: 4) {
                    Text("Avg. LV Coverage")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("\(String(format: "%.1f", averageLVPercentage))%")
                        .font(.title2)
                        .fontWeight(.semibold)
                }
                
                Spacer()
                
                VStack(alignment: .trailing, spacing: 4) {
                    Text("Quality")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(qualityAssessment)
                        .font(.title2)
                        .fontWeight(.semibold)
                        .foregroundColor(qualityColor)
                }
            }
        }
        .padding()
        .background(Color(.secondarySystemGroupedBackground))
        .cornerRadius(12)
    }
    
    private var qualityColor: Color {
        switch qualityAssessment {
        case "Excellent": return .green
        case "Good": return .blue
        case "Fair": return .orange
        default: return .red
        }
    }
}
