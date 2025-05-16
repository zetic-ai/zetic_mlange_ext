package com.zeticai.mlange.feature.whisper

class WhisperWrapper(
    vocabularyPath: String
) {
    init {
        nativeInit(vocabularyPath)
    }

    fun process(audio: FloatArray): FloatArray {
        return nativeProcess(audio)
    }

    fun decodeToken(ids: IntArray, skipSpecialToken: Boolean): String {
        return nativeDecodeToken(ids, skipSpecialToken)
    }

    fun deinit() {
        nativeDeinit()
    }

    private external fun nativeInit(vocabularyPath: String)

    private external fun nativeDeinit()

    private external fun nativeProcess(audio: FloatArray): FloatArray

    private external fun nativeDecodeToken(ids: IntArray, skipSpecialToken: Boolean): String

    companion object {
        init {
            System.loadLibrary("whisper_jni_wrapper")
        }
    }
}