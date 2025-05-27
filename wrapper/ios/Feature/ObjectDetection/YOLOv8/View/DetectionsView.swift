import SwiftUI

public struct DetectionsView : View {
    private let detectionResult: ObjectDetectionResult
    private let cameraResolution: CGSize
    
    private let classes = CocoConfig.readClasses()
    
    public init(detectionResult: ObjectDetectionResult, cameraResolution: CGSize) {
        self.detectionResult = detectionResult
        self.cameraResolution = cameraResolution
    }
    
    public var body: some View {
        GeometryReader { geometry in
            let aspectRatio: CGFloat = geometry.size.height / geometry.size.width
            ForEach(Array(zip(detectionResult.value.indices, detectionResult.value)), id: \.0) { index, result in
                let color = getClassColor(classId: result.classId)
                let confidence = String(format: "%.2f", result.confidence)
                let label = "\(classes[Int(result.classId)]) \(confidence)"
                let targetSize = CGSize(width: geometry.size.width, height: geometry.size.width * aspectRatio)
                let box = calculateBox(for: result, in: targetSize, cameraResolution)
                
                DetectionBoxView(
                    color: color,
                    label: label,
                    box: box)
                .customFrame(targetWidth: targetSize.width, targetHeight: targetSize.height,
                             geometryWidth: geometry.size.width, geometryHeight: geometry.size.height)
            }
        }
    }
    
    private func getClassColor(classId: Int32) -> Color {
        let r = Double((Int(classId) + 72) * 1717 % 256) / 255.0
        let g = Double((Int(classId) + 7) * 33 % 126 + 70) / 255.0
        let b = Double((Int(classId) + 47) * 107 % 256) / 255.0
        return Color(red: r, green: g, blue: b)
    }
    
    private func calculateBox(for result: YOLOv8Result, in targetSize: CGSize, _ cameraResolution: CGSize) -> [CGFloat] {
        let ret = [
            CGFloat(result.box[0]) * (targetSize.width / cameraResolution.height),
            CGFloat(result.box[1]) * (targetSize.height / cameraResolution.width),
            CGFloat(result.box[2]) * (targetSize.width / cameraResolution.height),
            CGFloat(result.box[3]) * (targetSize.height / cameraResolution.width)
        ]
        return ret
    }
}
