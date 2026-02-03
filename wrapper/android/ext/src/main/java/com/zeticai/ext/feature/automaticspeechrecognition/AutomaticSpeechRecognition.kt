package com.zeticai.ext.feature.automaticspeechrecognition

import android.content.Context
import com.zeticai.ext.feature.AutoSelectPipelineFeature
import com.zeticai.ext.feature.PipelineFeature
import com.zeticai.ext.feature.automaticspeechrecognition.whisper.Whisper

class AutomaticSpeechRecognition(
        private val context: Context
    ) : AutoSelectPipelineFeature<String>() {
        override fun selectFeature(): PipelineFeature<String> {
            return Whisper(context, "")
        }
    }
