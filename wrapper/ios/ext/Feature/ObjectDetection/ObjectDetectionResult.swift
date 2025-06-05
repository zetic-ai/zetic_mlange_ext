public class ObjectDetectionResult {
    let value: [YOLOv8Result]
    
    public init(_ value: [YOLOv8Result]) {
        self.value = value
    }
}
