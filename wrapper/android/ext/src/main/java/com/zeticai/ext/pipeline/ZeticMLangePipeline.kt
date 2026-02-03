package com.zeticai.ext.pipeline

import com.zeticai.ext.feature.PipelineFeature
import com.zeticai.ext.inputsource.InputSource

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
