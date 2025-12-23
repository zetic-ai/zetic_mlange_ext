package com.zeticai.mlange.feature.texttospeech.vits

class VitsWrapper(
    vocabularyPath: String
) {
    init {
        nativeInit(vocabularyPath)
    }

    fun convertTextToIds(text: String, maxLength: Int): Pair<IntArray, IntArray> {
        return nativeConvertTextToIds(text, maxLength)
    }


    fun deinit() {
        nativeDeinit()
    }

    private external fun nativeInit(vocabularyPath: String)

    private external fun nativeDeinit()

    private external fun nativeConvertTextToIds(
        text: String,
        maxLength: Int
    ): Pair<IntArray, IntArray>

    companion object {
        init {
            System.loadLibrary("vits_jni_wrapper")
        }
    }
}