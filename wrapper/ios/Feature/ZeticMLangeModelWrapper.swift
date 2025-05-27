import Foundation
import ZeticMLange

class ZeticMLangeModelWrapper: PipelineInferenceModel {
    private let model: ZeticMLangeModel
    
    init(_ tokenKey: String, _ modelKey: String) {
        self.model = (try? ZeticMLangeModel(tokenKey, modelKey))!
    }
    
    func inference(_ input: [Data]) -> [Data] {
        do {
            try model.run(input)
            return model.getOutputDataArray()
        } catch {
            return []
        }
    }
}
