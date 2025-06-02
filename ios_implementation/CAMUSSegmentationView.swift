import SwiftUI
import PhotosUI

/**
 * Main SwiftUI view for CAMUS Left Ventricle Segmentation
 * 
 * This view provides a complete interface for:
 * - Image selection from photo library or camera
 * - Real-time segmentation processing
 * - Results visualization with overlays
 * - Performance monitoring and statistics
 */
struct CAMUSSegmentationView: View {
    
    // MARK: - State Properties
    
    @StateObject private var viewModel = SegmentationViewModel()
    @State private var selectedPhotoItem: PhotosPickerItem?
    @State private var inputImage: UIImage?
    @State private var showingImagePicker = false
    @State private var showingCamera = false
    @State private var showingPerformanceStats = false
    @State private var overlayOpacity: Double = 0.6
    @State private var showOverlay = true
    
    // MARK: - Body
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    
                    // Header
                    headerSection
                    
                    // Image Display Section
                    imageDisplaySection
                    
                    // Controls Section
                    controlsSection
                    
                    // Results Section
                    if let result = viewModel.currentResult {
                        resultsSection(result: result)
                    }
                    
                    // Performance Section
                    performanceSection
                    
                    Spacer(minLength: 50)
                }
                .padding()
            }
            .navigationTitle("LV Segmentation")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Stats") {
                        showingPerformanceStats = true
                    }
                }
            }
        }
        .onChange(of: selectedPhotoItem) { newItem in
            loadSelectedImage(newItem)
        }
        .alert("Error", isPresented: $viewModel.showingError) {
            Button("OK") { }
        } message: {
            Text(viewModel.errorMessage)
        }
        .sheet(isPresented: $showingPerformanceStats) {
            PerformanceStatsView()
        }
        .onAppear {
            // Warm up model on app launch
            viewModel.warmUpModel()
        }
    }
    
    // MARK: - View Sections
    
    private var headerSection: some View {
        VStack(spacing: 8) {
            Image(systemName: "heart.circle.fill")
                .font(.system(size: 40))
                .foregroundColor(.red)
            
            Text("Left Ventricle Segmentation")
                .font(.title2)
                .fontWeight(.semibold)
            
            Text("Upload a cardiac ultrasound image for AI-powered left ventricle analysis")
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(15)
        .shadow(color: .black.opacity(0.1), radius: 5, x: 0, y: 2)
    }
    
    private var imageDisplaySection: some View {
        VStack(spacing: 15) {
            
            // Main Image Display
            Group {
                if let displayImage = getDisplayImage() {
                    Image(uiImage: displayImage)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(maxHeight: 300)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                        .overlay(
                            RoundedRectangle(cornerRadius: 12)
                                .stroke(Color.gray.opacity(0.3), lineWidth: 1)
                        )
                } else {
                    placeholderImageView
                }
            }
            
            // Overlay Controls (if segmentation exists)
            if viewModel.currentResult?.hasLeftVentricle == true {
                overlayControlsView
            }
        }
    }
    
    private var placeholderImageView: some View {
        RoundedRectangle(cornerRadius: 12)
            .fill(Color.gray.opacity(0.2))
            .frame(height: 250)
            .overlay(
                VStack(spacing: 12) {
                    Image(systemName: "photo.on.rectangle.angled")
                        .font(.system(size: 40))
                        .foregroundColor(.gray)
                    
                    Text("Select an ultrasound image")
                        .font(.headline)
                        .foregroundColor(.gray)
                    
                    Text("Choose from photos or take a new image")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            )
    }
    
    private var overlayControlsView: some View {
        VStack(spacing: 10) {
            HStack {
                Toggle("Show Overlay", isOn: $showOverlay)
                    .toggleStyle(SwitchToggleStyle())
                
                Spacer()
                
                Text("Opacity")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Slider(value: $overlayOpacity, in: 0.1...1.0, step: 0.1)
                    .frame(width: 100)
            }
            
            // Legend
            HStack(spacing: 15) {
                legendItem(color: .red, label: "LV Cavity")
                legendItem(color: .green, label: "LV Wall")
                Spacer()
            }
            .font(.caption)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(10)
    }
    
    private func legendItem(color: Color, label: String) -> some View {
        HStack(spacing: 5) {
            Circle()
                .fill(color)
                .frame(width: 10, height: 10)
            Text(label)
                .foregroundColor(.secondary)
        }
    }
    
    private var controlsSection: some View {
        VStack(spacing: 15) {
            
            // Image Selection Buttons
            HStack(spacing: 15) {
                
                // Photo Library Button
                PhotosPicker(
                    selection: $selectedPhotoItem,
                    matching: .images,
                    photoLibrary: .shared()
                ) {
                    Label("Photo Library", systemImage: "photo.on.rectangle")
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.blue)
                        .cornerRadius(10)
                }
                
                // Camera Button
                Button(action: { showingCamera = true }) {
                    Label("Camera", systemImage: "camera")
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.green)
                        .cornerRadius(10)
                }
            }
            
            // Test Image Button
            Button("Use Test Image") {
                inputImage = UIImage.createTestUltrasoundImage()
            }
            .foregroundColor(.blue)
            .padding(.vertical, 8)
            
            // Analyze Button
            Button(action: analyzeImage) {
                HStack {
                    if viewModel.isProcessing {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: .white))
                            .scaleEffect(0.8)
                    } else {
                        Image(systemName: "heart.text.square")
                    }
                    
                    Text(viewModel.isProcessing ? "Analyzing..." : "Analyze Left Ventricle")
                }
                .foregroundColor(.white)
                .frame(maxWidth: .infinity)
                .padding()
                .background(
                    (inputImage != nil && !viewModel.isProcessing) 
                        ? Color.red 
                        : Color.gray
                )
                .cornerRadius(10)
            }
            .disabled(inputImage == nil || viewModel.isProcessing)
        }
    }
    
    private func resultsSection(result: SegmentationResult) -> some View {
        VStack(alignment: .leading, spacing: 15) {
            
            Text("Analysis Results")
                .font(.title2)
                .fontWeight(.semibold)
            
            // Quick Stats Cards
            LazyVGrid(columns: [
                GridItem(.flexible()),
                GridItem(.flexible())
            ], spacing: 10) {
                
                statCard(
                    title: "Inference Time",
                    value: "\(String(format: "%.0f", result.inferenceTime * 1000)) ms",
                    icon: "speedometer",
                    color: .blue
                )
                
                statCard(
                    title: "Confidence",
                    value: "\(String(format: "%.1f", result.confidence * 100))%",
                    icon: "checkmark.seal",
                    color: .green
                )
                
                statCard(
                    title: "Quality Score",
                    value: "\(String(format: "%.0f", result.qualityScore * 100))%",
                    icon: "star.fill",
                    color: .orange
                )
                
                statCard(
                    title: "LV Coverage",
                    value: "\(String(format: "%.1f", result.leftVentriclePercentage))%",
                    icon: "heart.fill",
                    color: .red
                )
            }
            
            // Detailed Results
            if result.hasLeftVentricle {
                detailedResultsView(result: result)
            } else {
                noDetectionView
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(15)
        .shadow(color: .black.opacity(0.1), radius: 5, x: 0, y: 2)
    }
    
    private func statCard(title: String, value: String, icon: String, color: Color) -> some View {
        VStack(spacing: 8) {
            Image(systemName: icon)
                .font(.title2)
                .foregroundColor(color)
            
            Text(value)
                .font(.headline)
                .fontWeight(.bold)
            
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(10)
    }
    
    private func detailedResultsView(result: SegmentationResult) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            
            HStack {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundColor(.green)
                Text("Left Ventricle Detected")
                    .font(.headline)
                    .fontWeight(.semibold)
                Spacer()
            }
            
            // Pixel Statistics
            VStack(alignment: .leading, spacing: 6) {
                Text("Segmentation Details")
                    .font(.subheadline)
                    .fontWeight(.medium)
                
                detailRow("Cavity pixels", "\(result.segmentationStats.cavityPixels)")
                detailRow("Wall pixels", "\(result.segmentationStats.wallPixels)")
                detailRow("Total LV pixels", "\(result.segmentationStats.leftVentriclePixels)")
                detailRow("Cavity/Wall ratio", String(format: "%.2f", result.cavityToWallRatio))
            }
            .padding()
            .background(Color.green.opacity(0.1))
            .cornerRadius(8)
        }
    }
    
    private var noDetectionView: some View {
        VStack(spacing: 10) {
            Image(systemName: "exclamationmark.triangle")
                .font(.title)
                .foregroundColor(.orange)
            
            Text("No Left Ventricle Detected")
                .font(.headline)
                .fontWeight(.semibold)
            
            Text("The model could not identify left ventricle structures in this image. Try with a clearer cardiac ultrasound image.")
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding()
        .background(Color.orange.opacity(0.1))
        .cornerRadius(8)
    }
    
    private func detailRow(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label)
                .font(.caption)
                .foregroundColor(.secondary)
            Spacer()
            Text(value)
                .font(.caption)
                .fontWeight(.medium)
        }
    }
    
    private var performanceSection: some View {
        Group {
            if let stats = PerformanceMonitor.getPerformanceStats() {
                VStack(alignment: .leading, spacing: 10) {
                    Text("Performance Summary")
                        .font(.headline)
                    
                    Text(stats.formattedSummary)
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    if !PerformanceMonitor.isPerformanceAcceptable() {
                        Text("⚠️ Performance below target. Tap 'Stats' for optimization suggestions.")
                            .font(.caption)
                            .foregroundColor(.orange)
                    }
                }
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(10)
            }
        }
    }
    
    // MARK: - Helper Methods
    
    private func getDisplayImage() -> UIImage? {
        if let result = viewModel.currentResult,
           let original = inputImage,
           result.hasLeftVentricle && showOverlay {
            
            // Create overlay and combine with original
            if let overlayImage = SegmentationVisualizer.createOverlayImage(
                from: result.segmentationMask,
                originalSize: original.size,
                alpha: Float(overlayOpacity)
            ) {
                return SegmentationVisualizer.combineImages(
                    original: original,
                    overlay: overlayImage,
                    overlayAlpha: overlayOpacity
                )
            }
        }
        
        return inputImage
    }
    
    private func loadSelectedImage(_ item: PhotosPickerItem?) {
        guard let item = item else { return }
        
        Task {
            do {
                if let data = try await item.loadTransferable(type: Data.self),
                   let uiImage = UIImage(data: data) {
                    DispatchQueue.main.async {
                        self.inputImage = uiImage
                        // Clear previous results when new image is loaded
                        self.viewModel.clearResults()
                    }
                }
            } catch {
                print("❌ Error loading selected image: \(error)")
            }
        }
    }
    
    private func analyzeImage() {
        guard let image = inputImage else { return }
        viewModel.processImage(image)
    }
}

// MARK: - View Model

/**
 * ViewModel for managing segmentation state and model interactions
 */
class SegmentationViewModel: ObservableObject {
    
    // MARK: - Published Properties
    
    @Published var isProcessing = false
    @Published var currentResult: SegmentationResult?
    @Published var showingError = false
    @Published var errorMessage = ""
    @Published var modelReady = false
    
    // MARK: - Private Properties
    
    private let model = CAMUSSegmentationModel()
    
    // MARK: - Initialization
    
    init() {
        checkModelReadiness()
    }
    
    // MARK: - Public Methods
    
    func processImage(_ image: UIImage) {
        guard model.isReady else {
            showError("Model not ready. Please wait for initialization to complete.")
            return
        }
        
        isProcessing = true
        
        model.predict(image: image) { [weak self] result in
            DispatchQueue.main.async {
                self?.isProcessing = false
                
                switch result {
                case .success(let segmentationResult):
                    self?.currentResult = segmentationResult
                    
                    // Record performance
                    PerformanceMonitor.recordInferenceTime(segmentationResult.inferenceTime)
                    
                case .failure(let error):
                    self?.showError(error.localizedDescription)
                    
                    // Clear results on error
                    self?.currentResult = nil
                }
            }
        }
    }
    
    func warmUpModel() {
        guard model.isReady else { return }
        
        DispatchQueue.global(qos: .background).async { [weak self] in
            self?.model.warmUp()
        }
    }
    
    func clearResults() {
        currentResult = nil
    }
    
    // MARK: - Private Methods
    
    private func checkModelReadiness() {
        // Check if model is ready
        modelReady = model.isReady
        
        if !modelReady {
            // Check device capability
            if !DeviceCapabilityChecker.isDeviceCapable() {
                showError("Device may not have sufficient resources for optimal performance.")
            }
        }
    }
    
    private func showError(_ message: String) {
        errorMessage = message
        showingError = true
    }
}

// MARK: - Performance Stats View

struct PerformanceStatsView: View {
    
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                
                if let stats = PerformanceMonitor.getPerformanceStats() {
                    
                    Text("Performance Statistics")
                        .font(.title2)
                        .fontWeight(.semibold)
                    
                    VStack(alignment: .leading, spacing: 15) {
                        
                        Text(stats.formattedSummary)
                            .font(.system(.body, design: .monospaced))
                            .padding()
                            .background(Color(.systemGray6))
                            .cornerRadius(8)
                        
                        // Performance Status
                        HStack {
                            Image(systemName: PerformanceMonitor.isPerformanceAcceptable() ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                                .foregroundColor(PerformanceMonitor.isPerformanceAcceptable() ? .green : .orange)
                            
                            Text(PerformanceMonitor.isPerformanceAcceptable() ? "Performance Acceptable" : "Performance Below Target")
                                .fontWeight(.medium)
                        }
                        
                        // Optimization Suggestions
                        let suggestions = PerformanceMonitor.getOptimizationSuggestions()
                        if !suggestions.isEmpty {
                            VStack(alignment: .leading, spacing: 8) {
                                Text("Optimization Suggestions:")
                                    .fontWeight(.medium)
                                
                                ForEach(suggestions, id: \.self) { suggestion in
                                    Text("• \(suggestion)")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                            }
                            .padding()
                            .background(Color.orange.opacity(0.1))
                            .cornerRadius(8)
                        }
                        
                        // Device Info
                        VStack(alignment: .leading, spacing: 5) {
                            Text("Device Information:")
                                .fontWeight(.medium)
                            
                            Text(DeviceCapabilityChecker.getDeviceInfo())
                                .font(.system(.caption, design: .monospaced))
                                .foregroundColor(.secondary)
                        }
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(8)
                    }
                    
                } else {
                    
                    VStack(spacing: 15) {
                        Image(systemName: "chart.line.uptrend.xyaxis")
                            .font(.system(size: 50))
                            .foregroundColor(.gray)
                        
                        Text("No Performance Data")
                            .font(.headline)
                        
                        Text("Run some inferences to see performance statistics")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                
                Spacer()
            }
            .padding()
            .navigationTitle("Performance")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}

// MARK: - Previews

struct CAMUSSegmentationView_Previews: PreviewProvider {
    static var previews: some View {
        CAMUSSegmentationView()
    }
}
