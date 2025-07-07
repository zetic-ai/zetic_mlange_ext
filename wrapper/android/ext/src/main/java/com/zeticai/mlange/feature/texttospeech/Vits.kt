package com.zeticai.mlange.feature.texttospeech.vits


import android.content.Context
import com.zeticai.mlange.feature.PipelineFeature
import com.zeticai.mlange.feature.PipelineInferenceModel
import com.zeticai.mlange.feature.ZeticMLangeModelWrapper
import com.zeticai.mlange.feature.automaticspeechrecognition.whisper.WhisperWrapper


class Vits(
    private val context: Context,
    vocabularyPath: String = "",
    private val vitsModel: PipelineInferenceModel = ZeticMLangeModelWrapper(context, "", ""),
) : PipelineFeature<String> {
    val wrapper = WhisperWrapper(vocabularyPath)

    override fun preprocess(input: Array<ByteArray>): Array<ByteArray> {
        TODO("Not yet implemented")
    }

    override fun postprocess(output: Array<ByteArray>): String {
        TODO("Not yet implemented")
    }

    override fun run(input: Array<ByteArray>): String {
        TODO("Not yet implemented")
    }
}