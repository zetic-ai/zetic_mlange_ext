package com.zeticai.mlange.feature.facedetection

import java.nio.ByteBuffer

class FaceDetectionWrapper {

    private val zeticMLangeFeatureFaceDetectionPtr: Long = nativeInit()

    private external fun nativeInit(): Long
    private external fun nativeDeinit(zeticMLangeFeatureFaceDetectionPtr: Long)

    private external fun nativePreprocess(
        zeticMLangeFeatureFaceDetectionPtr: Long,
        inputImgPtr: Long
    ): ByteArray

    private external fun nativePostprocess(
        zeticMLangeFeatureFaceDetectionPtr: Long,
        outputData: Array<ByteArray>
    ): FaceDetectionResults

    fun preprocess(imagePtr: Long): ByteBuffer {
        val output = nativePreprocess(zeticMLangeFeatureFaceDetectionPtr, imagePtr)
        return ByteBuffer.wrap(output)
    }

    fun postprocess(outputData: Array<ByteArray>): FaceDetectionResults {
        return nativePostprocess(zeticMLangeFeatureFaceDetectionPtr, outputData)
    }

    fun deinit() {
        nativeDeinit(zeticMLangeFeatureFaceDetectionPtr)
    }

    companion object {
        init {
            System.loadLibrary("face_detection_jni_wrapper")
        }
    }
}
