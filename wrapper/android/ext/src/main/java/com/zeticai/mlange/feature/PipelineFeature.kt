package com.zeticai.mlange.feature

interface PipelineFeature<T> {
    fun preprocess(input: Array<ByteArray>): Array<ByteArray>
    fun postprocess(output: Array<ByteArray>): T
    fun run(input: Array<ByteArray>): T
}