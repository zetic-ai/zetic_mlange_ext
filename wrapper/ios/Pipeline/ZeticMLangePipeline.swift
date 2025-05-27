import Foundation
import Combine

public class ZeticMLangePipeline<Input, Output>: ObservableObject {
    private let feature: any PipelineFeature<Input, Output>
    private let inputSource: (any InputSource<Input>)?
    
    @Published public var latestResult: Output?
    
    public init(
        feature: any PipelineFeature<Input, Output>,
        inputSource: (any InputSource<Input>)? = nil
    ) {
        self.feature = feature
        self.inputSource = inputSource
    }
    
    public func run(input: Input) -> Output? {
        return feature.run(input)
    }
    
    public func startLoop() {
        inputSource?.acquire { [weak self] input in
            DispatchQueue.main.async {
                self?.latestResult = self?.run(input: input)
            }
        }
    }
    
    public func stopLoop() {
        inputSource?.stop()
    }
}
