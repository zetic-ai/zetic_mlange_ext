import Foundation
import Combine

open class AsyncFeature<Input: AsyncFeatureInput, Output: AsyncFeatureOutput>: ObservableObject {
    enum Status {
        case WAITING
        case RUNNING
        case CLOSED
        case PAUSED
    }
    
    private var status: Status = Status.WAITING
    private let operationQueue: DispatchQueue
    private let operationGroup = DispatchGroup()
    
    public init(label: String) {
        self.operationQueue = DispatchQueue(label: label)
    }
    
    open func process(input: Input) -> Output {
        fatalError("Subclasses must implement process(input:)")
    }
    
    open func handleOutput(_ output: Output) {
        fatalError("Subclasses must implement handleOutput(_:)")
    }
    
    public func run(with input: Input) {
        guard status == .WAITING else { return }
        
        status = .RUNNING
        operationGroup.enter()
        
        operationQueue.async { [weak self] in
            guard let self = self, self.status == .RUNNING else {
                self?.operationGroup.leave()
                return
            }
            
            let output = self.process(input: input)
            
            DispatchQueue.main.async {
                guard self.status == .RUNNING else {
                    self.operationGroup.leave()
                    return
                }
                self.handleOutput(output)
                self.status = .WAITING
                self.operationGroup.leave()
            }
        }
    }
    
    public func waitForPendingOperations(completion: @escaping () -> Void) {
        operationQueue.async { [weak self] in
            self?.operationGroup.wait()
            DispatchQueue.main.async {
                completion()
            }
        }
    }
    
    open func close() {
        status = .CLOSED
    }
    
    public func resume() {
        status = .WAITING
    }
    
    public func pause() {
        status = .PAUSED
    }
}

public protocol AsyncFeatureInput {}
public protocol AsyncFeatureOutput {}
