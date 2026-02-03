package com.zeticai.ext.feature

import com.zeticai.mlange.core.tensor.Tensor

fun interface PipelineInferenceModel {
    fun inference(inputs: Array<Tensor>): Array<Tensor>
}
