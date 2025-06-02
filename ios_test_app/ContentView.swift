import SwiftUI
import AVFoundation
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var segmentationProcessor = UltrasoundSegmentationProcessor()
    @State private var showingFilePicker = false
    @State private var selectedVideoURL: URL?
    @State private var isProcessing = false
    @State private var processingProgress: Double = 0.0
    @State private var showingResults = false
    @State private var errorMessage: String?
    
    var body: some View {
        NavigationView {
            VStack(spacing: 30) {
                // Header
                VStack(spacing: 10) {
                    Image(systemName: "heart.fill")
                        .font(.system(size: 60))
                        .foregroundColor(.red)
                    
                    Text("CAMUS Cardiac Segmentation")
                        .font(.title)
                        .fontWeight(.bold)
                    
                    Text("Upload 4-chamber ultrasound video for AI analysis")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                }
                .padding(.top, 50)
                
                Spacer()
                
                // Main content area
                if selectedVideoURL == nil {
                    // File selection area
                    VStack(spacing: 20) {
                        Button(action: {
                            showingFilePicker = true
                        }) {
                            VStack(spacing: 15) {
                                Image(systemName: "video.badge.plus")
                                    .font(.system(size: 50))
                                    .foregroundColor(.blue)
                                
                                Text("Select Ultrasound Video")
                                    .font(.headline)
                                    .foregroundColor(.blue)
                                
                                Text("Choose a 4-chamber ultrasound .mp4 file")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            .frame(maxWidth: .infinity)
                            .frame(height: 150)
                            .background(
                                RoundedRectangle(cornerRadius: 15)
                                    .stroke(Color.blue, style: StrokeStyle(lineWidth: 2, dash: [10]))
                            )
                        }
                        .padding(.horizontal, 40)
                        
                        Text("Supported formats: MP4, MOV")
                            .font(.caption2)
                            .foregroundColor(.tertiary)
                    }
                } else if isProcessing {
                    // Processing view
                    VStack(spacing: 20) {
                        ProgressView(value: processingProgress, total: 1.0)
                            .progressViewStyle(LinearProgressViewStyle())
                            .frame(width: 250)
                        
                        Text("Processing Video...")
                            .font(.headline)
                        
                        Text("\(Int(processingProgress * 100))% Complete")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                        
                        Text(segmentationProcessor.currentStatus)
                            .font(.caption)
                            .foregroundColor(.blue)
                            .multilineTextAlignment(.center)
                    }
                    .padding()
                } else {
                    // Video selected, ready to process
                    VStack(spacing: 20) {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.system(size: 50))
                            .foregroundColor(.green)
                        
                        Text("Video Selected")
                            .font(.headline)
                        
                        if let url = selectedVideoURL {
                            Text(url.lastPathComponent)
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                        
                        Button("Start Segmentation Analysis") {
                            startProcessing()
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.large)
                        
                        Button("Choose Different Video") {
                            selectedVideoURL = nil
                        }
                        .buttonStyle(.bordered)
                    }
                }
                
                // Error message
                if let error = errorMessage {
                    Text(error)
                        .foregroundColor(.red)
                        .font(.caption)
                        .padding()
                        .background(Color.red.opacity(0.1))
                        .cornerRadius(8)
                        .padding(.horizontal)
                }
                
                Spacer()
                
                // Footer info
                VStack(spacing: 5) {
                    Text("AI-powered left ventricle segmentation")
                        .font(.caption2)
                        .foregroundColor(.tertiary)
                    
                    Text("Processing time: ~2-5 seconds per frame")
                        .font(.caption2)
                        .foregroundColor(.tertiary)
                }
                .padding(.bottom, 30)
            }
            .navigationTitle("")
            .navigationBarHidden(true)
        }
        .fileImporter(
            isPresented: $showingFilePicker,
            allowedContentTypes: [.movie, .quickTimeMovie, .mpeg4Movie],
            allowsMultipleSelection: false
        ) { result in
            handleFileSelection(result)
        }
        .fullScreenCover(isPresented: $showingResults) {
            if let videoURL = selectedVideoURL {
                SegmentationResultsView(
                    videoURL: videoURL,
                    segmentationResults: segmentationProcessor.segmentationResults,
                    onDismiss: {
                        showingResults = false
                        resetToInitialState()
                    }
                )
            }
        }
        .onReceive(segmentationProcessor.$processingProgress) { progress in
            self.processingProgress = progress
        }
        .onReceive(segmentationProcessor.$isProcessingComplete) { isComplete in
            if isComplete && !segmentationProcessor.segmentationResults.isEmpty {
                self.isProcessing = false
                self.showingResults = true
            }
        }
        .onReceive(segmentationProcessor.$errorMessage) { error in
            if let error = error {
                self.errorMessage = error
                self.isProcessing = false
            }
        }
    }
    
    private func handleFileSelection(_ result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            if let url = urls.first {
                selectedVideoURL = url
                errorMessage = nil
                
                // Validate the video file
                validateVideoFile(url)
            }
        case .failure(let error):
            errorMessage = "Failed to select file: \(error.localizedDescription)"
        }
    }
    
    private func validateVideoFile(_ url: URL) {
        let asset = AVAsset(url: url)
        
        Task {
            do {
                let duration = try await asset.load(.duration)
                let tracks = try await asset.load(.tracks)
                
                await MainActor.run {
                    if duration.seconds < 0.5 {
                        errorMessage = "Video is too short. Please select a video with at least 0.5 seconds."
                    } else if duration.seconds > 30 {
                        errorMessage = "Video is too long. Please select a video shorter than 30 seconds for optimal processing."
                    } else if tracks.isEmpty {
                        errorMessage = "Invalid video file. No video tracks found."
                    } else {
                        errorMessage = nil
                    }
                }
            } catch {
                await MainActor.run {
                    errorMessage = "Failed to validate video: \(error.localizedDescription)"
                }
            }
        }
    }
    
    private func startProcessing() {
        guard let videoURL = selectedVideoURL else { return }
        
        isProcessing = true
        errorMessage = nil
        
        segmentationProcessor.processVideo(url: videoURL)
    }
    
    private func resetToInitialState() {
        selectedVideoURL = nil
        isProcessing = false
        processingProgress = 0.0
        errorMessage = nil
        segmentationProcessor.reset()
    }
}
