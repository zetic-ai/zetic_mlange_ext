public struct FaceEmotionRecognitionResult {
    public let emotion: String
    public let confidence: Float
    
    public init(emotion: String, confidence: Float) {
        self.emotion = emotion
        self.confidence = confidence
    }
}
