package com.zeticai.mlange.feature

fun interface PipelineInferenceModel {
    fun inference(input: Array<ByteArray>): Array<ByteArray>
}
