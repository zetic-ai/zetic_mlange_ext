package com.zeticai.mlange.feature

import android.content.Context
import com.zeticai.mlange.core.model.ZeticMLangeModel
import com.zeticai.mlange.core.tensor.Tensor

class ZeticMLangeModelWrapper(
    context: Context,
    personalKey: String,
    modelKey: String
) : PipelineInferenceModel {
    private val model = ZeticMLangeModel(context, personalKey, modelKey)

    override fun inference(inputs: Array<Tensor>): Array<Tensor> {
        return model.run(inputs)
    }
}
