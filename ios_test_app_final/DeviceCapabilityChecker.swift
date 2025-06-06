import Foundation

class DeviceCapabilityChecker {
    
    static func getRecommendedThreadCount() -> Int {
        // Simple implementation that returns the number of physical cores
        // minus 1 (to leave cores for the system)
        let physicalCores = ProcessInfo.processInfo.processorCount
        return max(1, physicalCores - 1)
    }
}