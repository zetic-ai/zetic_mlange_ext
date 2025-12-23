package com.zeticai.mlange.feature

import java.nio.ByteBuffer

interface PipelineFeature<T> {
    fun preprocess(input: Array<ByteArray>): Array<ByteArray>
    fun postprocess(output: Array<ByteBuffer>): T
    fun run(input: Array<ByteArray>): T
}