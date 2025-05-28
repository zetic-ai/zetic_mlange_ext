package com.zeticai.mlange.feature.faceemotionrecognition

import com.zeticai.mlange.feature.entity.Box
import java.nio.ByteBuffer

class FaceEmotionRecognitionWrapper {

    private val zeticMLangeFeatureFaceEmotionRecognitionPtr: Long = nativeInit()

    private external fun nativeInit(): Long
    private external fun nativeDeinit(zeticMLangeFeatureFaceEmotionRecognitionPtr: Long)

    private external fun nativePreprocess(
        zeticMLangeFeatureFaceEmotionRecognitionPtr: Long,
        inputImgPtr: Long,
        xmin: Float,
        ymin: Float,
        xmax: Float,
        ymax: Float
    ): ByteArray

    private external fun nativePostprocess(
        zeticMLangeFeatureFaceEmotionRecognitionPtr: Long,
        outputData: Array<ByteArray>
    ): FaceEmotionRecognitionResult

    fun preprocess(imagePtr: Long, roi: Box): ByteBuffer {
        return ByteBuffer.wrap(
            nativePreprocess(zeticMLangeFeatureFaceEmotionRecognitionPtr, imagePtr, roi.xMin, roi.yMin, roi.xMax, roi.yMax)
        )
    }

    fun postprocess(outputData: Array<ByteArray>): FaceEmotionRecognitionResult {
        return nativePostprocess(zeticMLangeFeatureFaceEmotionRecognitionPtr, outputData)
    }

    fun deinit() {
        nativeDeinit(zeticMLangeFeatureFaceEmotionRecognitionPtr)
    }

    companion object {
        init {
            System.loadLibrary("face_emotion_recognition_jni_wrapper")
        }
    }
}
