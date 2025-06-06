import SwiftUI
import AVFoundation

struct SegmentationResultsView: View {
    let videoURL: URL
    let segmentationResults: [FrameSegmentationResult]
    let onDismiss: () -> Void
    
    @State private var currentFrameIndex: Int = 0
    @State private var isPlaying: Bool = false
    @State private var playbackTimer: Timer?
    @State private var selectedViewMode: ViewMode = .sideBySide
    @State private var playbackSpeed: Double = 1.0
    
    enum ViewMode: String, CaseIterable {
        case original = "Original"
        case segmented = "Segmentado"
        case sideBySide = "Lado a Lado"
        case overlay = "Sobreposição"
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
                videoDisplaySection
                controlsSection
                if !segmentationResults.isEmpty {
                    summarySection
                }
            }
            .navigationTitle("Resultados da Segmentação")
            .navigationBarTitleDisplayMode(.inline)
            .navigationBarBackButtonHidden(true)
            .navigationBarItems(
                leading: Button("Concluído") {
                    stopPlayback()
                    onDismiss()
                },
                trailing: Button(action: {
                    shareResults()
                }) {
                    Image(systemName: "square.and.arrow.up")
                }
            )
        }
        .onDisappear {
            stopPlayback()
        }
    }
    
    private var videoDisplaySection: some View {
        ZStack {
            Color.black
                .aspectRatio(16/9, contentMode: .fit)
            
            if let frame = currentFrame {
                frameImageView(frame)
            } else {
                noFrameView
            }
            
            overlayControls
        }
    }
    
    private var noFrameView: some View {
        VStack {
            Image(systemName: "exclamationmark.triangle")
                .font(.system(size: 40))
                .foregroundColor(.yellow)
            Text("Nenhum dado de quadro disponível")
                .foregroundColor(.white)
        }
    }
    
    private var overlayControls: some View {
        VStack {
            HStack {
                Spacer()
                viewModeMenu
            }
            .padding()
            
            Spacer()
            
            if let frame = currentFrame {
                frameInfoOverlay(frame)
            }
        }
    }
    
    private var viewModeMenu: some View {
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
    }
    
    private func frameImageView(_ frame: FrameSegmentationResult) -> some View {
        Group {
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
                sideBySideView(frame)
            case .overlay:
                Image(uiImage: frame.visualizedImage)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
            }
        }
    }
    
    private func sideBySideView(_ frame: FrameSegmentationResult) -> some View {
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
    }
    
    private func frameInfoOverlay(_ frame: FrameSegmentationResult) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Quadro \(frame.frameIndex + 1)/\(segmentationResults.count)")
                    .font(.caption)
                    .fontWeight(.medium)
                
                Spacer()
                
                Text(frame.formattedTimestamp)
                    .font(.caption)
                    .fontWeight(.medium)
            }
            
            HStack {
                Label("VE: \(String(format: "%.1f", frame.segmentationResult.leftVentriclePercentage))%", 
                      systemImage: "heart.fill")
                    .font(.caption2)
                
                Spacer()
                
                Label("Confiança: \(String(format: "%.0f", frame.segmentationResult.confidence * 100))%", 
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
    
    private var controlsSection: some View {
        VStack(spacing: 20) {
            sliderSection
            buttonSection
            speedSection
        }
        .padding(.vertical)
        .background(Color(.systemGroupedBackground))
    }
    
    private var sliderSection: some View {
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
                Text("Quadro \(currentFrameIndex + 1)")
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
    }
    
    private var buttonSection: some View {
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
    }
    
    private var speedSection: some View {
        HStack {
            Text("Velocidade:")
                .font(.caption)
            
            Picker("Velocidade", selection: $playbackSpeed) {
                Text("0,5x").tag(0.5)
                Text("1x").tag(1.0)
                Text("2x").tag(2.0)
                Text("4x").tag(4.0)
            }
            .pickerStyle(.segmented)
        }
        .padding(.horizontal)
    }
    
    private var summarySection: some View {
        AnalysisSummaryView(results: segmentationResults)
            .padding()
            .background(Color(.systemGroupedBackground))
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
        
        let frameInterval = 1.0 / (5.0 * playbackSpeed)
        
        playbackTimer = Timer.scheduledTimer(withTimeInterval: frameInterval, repeats: true) { _ in
            if currentFrameIndex >= segmentationResults.count - 1 {
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
        guard !segmentationResults.isEmpty else {
            return """
            Análise de Segmentação Cardíaca CAMUS
            
            Vídeo: \(videoURL.lastPathComponent)
            Quadros Analisados: 0
            Confiança Média: N/A
            Cobertura VE Média: N/A
            
            Gerado pelo App de Segmentação CAMUS
            """
        }

        let confidenceValues = segmentationResults.map { Double($0.segmentationResult.confidence) }
        let lvPercentageValues = segmentationResults.map { Double($0.segmentationResult.leftVentriclePercentage) }
        
        let avgConfidence = confidenceValues.reduce(0, +) / Double(segmentationResults.count)
        let avgLVPercentage = lvPercentageValues.reduce(0, +) / Double(segmentationResults.count)
        
        return """
        Análise de Segmentação Cardíaca CAMUS
        
        Vídeo: \(videoURL.lastPathComponent)
        Quadros Analisados: \(segmentationResults.count)
        Confiança Média: \(String(format: "%.1f", avgConfidence * 100))%
        Cobertura VE Média: \(String(format: "%.1f", avgLVPercentage))%
        
        Gerado pelo App de Segmentação CAMUS
        """
    }
    
    private func shareResults() {
        let summaryText = generateSummaryText()
        let activityViewController = UIActivityViewController(
            activityItems: [summaryText],
            applicationActivities: nil
        )
        
        if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
           let window = windowScene.windows.first,
           let rootViewController = window.rootViewController {
            rootViewController.present(activityViewController, animated: true)
        }
    }
}

struct AnalysisSummaryView: View {
    let results: [FrameSegmentationResult]
    
    private var averageConfidence: Float {
        guard !results.isEmpty else { return 0.0 }
        let confidenceValues = results.map { $0.segmentationResult.confidence }
        let total = confidenceValues.reduce(0, +)
        return total / Float(results.count)
    }
    
    private var averageLVPercentage: Float {
        guard !results.isEmpty else { return 0.0 }
        let lvValues = results.map { $0.segmentationResult.leftVentriclePercentage }
        let total = lvValues.reduce(0, +)
        return total / Float(results.count)
    }
    
    private var qualityAssessment: String {
        if averageConfidence > 0.8 && averageLVPercentage > 15 {
            return "Excelente"
        } else if averageConfidence > 0.6 && averageLVPercentage > 10 {
            return "Bom"
        } else if averageConfidence > 0.4 && averageLVPercentage > 5 {
            return "Regular"
        } else {
            return "Ruim"
        }
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Resumo da Análise")
                .font(.headline)
            
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Quadros Analisados")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("\(results.count)")
                        .font(.title2)
                        .fontWeight(.semibold)
                }
                
                Spacer()
                
                VStack(alignment: .trailing, spacing: 4) {
                    Text("Confiança Média")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("\(String(format: "%.0f", averageConfidence * 100))%")
                        .font(.title2)
                        .fontWeight(.semibold)
                }
                
                Spacer()
                
                VStack(alignment: .trailing, spacing: 4) {
                    Text("Cobertura VE Média")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("\(String(format: "%.1f", averageLVPercentage))%")
                        .font(.title2)
                        .fontWeight(.semibold)
                }
                
                Spacer()
                
                VStack(alignment: .trailing, spacing: 4) {
                    Text("Qualidade")
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
        case "Excelente": return .green
        case "Bom": return .blue
        case "Regular": return .orange
        default: return .red
        }
    }
}
