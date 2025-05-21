import AVFoundation
import ZeticMLange

class DeferredModel<Input: AsyncFeatureInput, Output: AsyncFeatureOutput> : AsyncFeature<Input, Output> {
    
    @Published var isInitialized: Bool = false
    internal var model: ZeticMLangeModel?
    
    init(label: String, modelKey: String, target: Target = .ZETIC_MLANGE_TARGET_COREML, onInitialized: (() -> Void)? = nil) {
        super.init(label: label)
        DispatchQueue.global(qos: .background).async {
            self.model = (try? ZeticMLangeModel(PrivateValues.personalKey, modelKey, target))!
            DispatchQueue.main.async {
                self.isInitialized = true
                onInitialized?()
            }
        }
    }
}
