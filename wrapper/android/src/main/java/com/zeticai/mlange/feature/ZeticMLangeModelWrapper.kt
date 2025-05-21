package com.zeticai.mlange.feature

import android.content.Context
import com.zeticai.mlange.core.model.ZeticMLangeModel

class ZeticMLangeModelWrapper(
    context: Context,
    personalKey: String,
    modelKey: String
) : PipelineInferenceModel {
    private val model = ZeticMLangeModel(context, personalKey, modelKey)

    override fun inference(input: Array<ByteArray>): Array<ByteArray> {
        model.run(input)
        return model.outputArrays
    }
}
