package com.zeticai.mlange.feature.automaticspeechrecognition.whisper

import android.content.Context
import com.zeticai.mlange.feature.PipelineFeature
import com.zeticai.mlange.feature.PipelineInferenceModel
import com.zeticai.mlange.feature.ZeticMLangeModelWrapper
import java.nio.ByteBuffer

class Whisper(
    private val context: Context,
    vocabularyPath: String = "",
    private val encoderModel: PipelineInferenceModel = ZeticMLangeModelWrapper(context, "", ""),
    private val decoderModel: PipelineInferenceModel = ZeticMLangeModelWrapper(context, "", ""),
) : PipelineFeature<String> {
    val wrapper = WhisperWrapper(vocabularyPath)

    override fun preprocess(input: Array<ByteArray>): Array<ByteArray> {
        TODO("Not yet implemented")
    }

    override fun postprocess(output: Array<ByteBuffer>): String {
        TODO("Not yet implemented")
    }

    override fun run(input: Array<ByteArray>): String {
        TODO("Not yet implemented")
    }
}
