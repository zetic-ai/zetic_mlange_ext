import Foundation

public protocol PipelineFeature<Input, Output> {
    associatedtype Input
    associatedtype Output
    func preprocess(_ input: Input) -> [Data]
    func postprocess(_ output: [Data]) -> Output
    func run(_ input: Input) -> Output
}
