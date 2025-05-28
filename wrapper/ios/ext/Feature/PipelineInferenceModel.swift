import Foundation

public protocol PipelineInferenceModel {
    func inference(_ input: [Data]) -> [Data]
}
