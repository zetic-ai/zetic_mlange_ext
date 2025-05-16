package com.zeticai.mlange.feature.facelandmark

import com.zeticai.mlange.feature.entity.Box
import java.nio.ByteBuffer

class FaceLandmarkWrapper {

    private val zeticMLangeFeatureFaceLandmarkPtr: Long = nativeInit()

    private external fun nativeInit(): Long
    private external fun nativeDeinit(zeticMLangeFeatureFaceLandmark: Long)

    private external fun nativePreprocess(
        zeticMLangeFeatureFaceLandmark: Long,
        imagePtr: Long,
        xmin: Float,
        ymin: Float,
        xmax: Float,
        ymax: Float
    ): ByteArray

    private external fun nativePostprocess(
        zeticMLangeFeatureFaceDetectionPtr: Long,
        outputData: Array<ByteArray>
    ): FaceLandmarkResult

    fun preprocess(imagePtr: Long, roi: Box): ByteBuffer {
        val output = nativePreprocess(zeticMLangeFeatureFaceLandmarkPtr, imagePtr, roi.xMin, roi.yMin, roi.xMax, roi.yMax)
        return ByteBuffer.wrap(output)
    }

    fun postprocess(outputData: Array<ByteArray>): FaceLandmarkResult {
        return nativePostprocess(zeticMLangeFeatureFaceLandmarkPtr, outputData)
    }

    fun deinit() {
        nativeDeinit(zeticMLangeFeatureFaceLandmarkPtr)
    }

    companion object {
        init {
            System.loadLibrary("face_landmark_jni_wrapper")
        }
    }
}
