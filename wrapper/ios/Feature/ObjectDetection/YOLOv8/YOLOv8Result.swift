public class YOLOv8Result {
    public let classId: Int32
    public let confidence: Float
    public let box: [Int32]
    
    public init(classId: Int32, confidence: Float, box: [Int32]) {
        self.classId = classId
        self.confidence = confidence
        self.box = box
    }
}
