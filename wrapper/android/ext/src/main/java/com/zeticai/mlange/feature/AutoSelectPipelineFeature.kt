package com.zeticai.mlange.feature

import java.nio.ByteBuffer

abstract class AutoSelectPipelineFeature<T> : PipelineFeature<T> {
    val feature: PipelineFeature<T> by lazy {
        selectFeature()
    }

    protected abstract fun selectFeature(): PipelineFeature<T>

    override fun preprocess(input: Array<ByteArray>): Array<ByteArray> {
        return feature.preprocess(input)
    }

    override fun postprocess(output: Array<ByteBuffer>): T {
        return feature.postprocess(output)
    }

    override fun run(input: Array<ByteArray>): T {
        return feature.run(input)
    }
}
