package com.zeticai.mlange.feature.vision

import android.view.Surface

class OpenCVImageUtilsWrapper {

    private external fun nativeSetSurface(
        surface: Surface
    )

    private external fun nativeFrame(
        image: ByteArray,
        rotate: Int
    ): Long

    fun setSurface(surface: Surface) {
        nativeSetSurface(surface)
    }

    fun frame(image: ByteArray, rotate: Int): Long {
        return nativeFrame(image, rotate)
    }

    companion object {
        init {
            System.loadLibrary("opencv_jni_wrapper")
        }
    }
}
