public struct FaceDetectionResult {
    public let bbox: Box
    public let confidence: Float
    
    public init(bbox: Box, confidence: Float) {
        self.bbox = bbox
        self.confidence = confidence
    }
}
