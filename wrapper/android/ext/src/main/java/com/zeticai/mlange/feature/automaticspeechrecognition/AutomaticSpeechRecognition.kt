package com.zeticai.mlange.feature.automaticspeechrecognition

import android.content.Context
import com.zeticai.mlange.feature.AutoSelectPipelineFeature
import com.zeticai.mlange.feature.PipelineFeature
import com.zeticai.mlange.feature.automaticspeechrecognition.whisper.Whisper

class AutomaticSpeechRecognition(
        private val context: Context
    ) : AutoSelectPipelineFeature<String>() {
        override fun selectFeature(): PipelineFeature<String> {
            return Whisper(context, "")
        }
    }
