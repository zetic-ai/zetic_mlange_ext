public class ObjectDetection : AutoSelectPipelineFeature<CameraFrame, ObjectDetectionResult> {
    public override init() {
        
    }
    
    override func selectFeature() -> any PipelineFeature<CameraFrame, ObjectDetectionResult> {
        return YOLOv8()
    }
}
