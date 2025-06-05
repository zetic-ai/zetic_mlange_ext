import Foundation

public class YOLOv8: PipelineFeature {
    
    private let wrapper: YOLOv8Wrapper
    private let model: PipelineInferenceModel
    
    public init(cocoYamlFilePath: String? = nil,
         model: PipelineInferenceModel? = nil) {
        
        let yamlURL = Bundle.main.url(forResource: "coco", withExtension: "yaml")!
        self.wrapper = YOLOv8Wrapper(yamlURL.absoluteString)
        
        self.model = model ?? ZeticMLangeModelWrapper(
            "ztp_97aT0F0HtHQ5Q3dasRCIAoxKH0O0YKJUyvOB",
            "b9f5d74e6f644288a32c50174ded828e"
        )
    }
    
    public func preprocess(_ input: CameraFrame) -> [Data] {
        return input.withUnsafeBytes { frameBytes in
            return [wrapper.featurePreprocess(
                frameBytes.baseAddress!,
                input.width,
                input.height,
                input.bytesPerRow
            )]
        }
    }
    
    public func postprocess(_ output: [Data]) -> ObjectDetectionResult {
        return output[0].withUnsafeBytes { (buffer: UnsafeRawBufferPointer) in
            let pointer = buffer.bindMemory(to: UInt8.self).baseAddress
            return if let uint8Pointer = pointer {
                ObjectDetectionResult(wrapper.featurePostprocess(uint8Pointer))
            } else {
                ObjectDetectionResult([])
            }
        }
    }
    
    public func run(_ input: CameraFrame) -> ObjectDetectionResult {
        let preprocessed = preprocess(input)
        let output = model.inference(preprocessed)
        return postprocess(output)
    }
}
