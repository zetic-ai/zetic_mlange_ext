public struct FaceLandmarkResult {
    public let faceLandmark: [Landmark]
    public let confidence: Float
    
    public init(faceLandmark: [Landmark], confidence: Float) {
        self.faceLandmark = faceLandmark
        self.confidence = confidence
    }
}

