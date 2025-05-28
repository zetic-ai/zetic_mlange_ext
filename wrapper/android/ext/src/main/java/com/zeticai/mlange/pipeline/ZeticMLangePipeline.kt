package com.zeticai.mlange.pipeline

import com.zeticai.mlange.feature.PipelineFeature
import com.zeticai.mlange.inputsource.InputSource

class ZeticMLangePipeline<T>(
    private val feature: PipelineFeature<T>,
    private val inputSource: InputSource? = null,
) {

    fun run(input: Array<ByteArray>): T {
        return feature.run(input)
    }

    fun loop(frame: (T) -> Unit) {
        inputSource?.acquire {
            val output = run(arrayOf(it))
            frame(output)
        }
    }
}
