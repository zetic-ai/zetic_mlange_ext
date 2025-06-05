import Foundation

public class AutoSelectPipelineFeature<Input, Output>: PipelineFeature {
    lazy var feature: any PipelineFeature<Input, Output> = selectFeature()
    
    func selectFeature() -> any PipelineFeature<Input, Output> {
        fatalError("Must override selectFeature()")
    }
    
    public func preprocess(_ input: Input) -> [Data] {
        return feature.preprocess(input)
    }
    
    public func postprocess(_ output: [Data]) -> Output {
        return feature.postprocess(output)
    }
    
    public func run(_ input: Input) -> Output {
        return feature.run(input)
    }
}
