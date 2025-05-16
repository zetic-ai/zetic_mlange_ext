package com.zeticai.mlange.feature.yolov8

import java.nio.ByteBuffer

class YOLOv8Wrapper(cocoYamlFilePath: String) {
    private val featurePtr: Long

    init {
        this.featurePtr = nativeInitDetect(cocoYamlFilePath)
    }

    fun preprocess(imagePtr: Long): ByteBuffer {
        val output = nativePreprocess(this.featurePtr, imagePtr)
        for (i in output.indices) {
            if (i % 4 == 0) output[i] = (output[i] - 1.toByte()).toByte()
        }
        return ByteBuffer.wrap(output)
    }

    fun postprocess(outputData: ByteArray): YOLOResult {
        return nativePostProcess(this.featurePtr, outputData)
    }

    fun preprocessWithFrame(
        frame: ByteArray,
        width: Int,
        height: Int,
        formatCode: Int
    ): ByteArray {
        return nativePreprocessWithFrame(
            this.featurePtr,
            frame,
            width,
            height,
            formatCode
        )
    }

    fun deinit() {
        nativeDeinit(this.featurePtr)
    }

    private external fun nativeInitDetect(cocoFilePath: String): Long

    private external fun nativeInitClassifier(cocoFilePath: String): Long

    private external fun nativePreprocess(featurePtr: Long, imagePtr: Long): ByteArray

    private external fun nativePostProcess(
        featurePtr: Long,
        outputData: ByteArray
    ): YOLOResult

    private external fun nativeFreePreprocessedBuffer(byteBuffer: ByteBuffer)

    private external fun nativeDeinit(featurePtr: Long)

    private external fun nativePreprocessWithFrame(
        zeticMLangeYolov8FeaturePtr: Long,
        frame: ByteArray,
        width: Int,
        height: Int,
        formatCode: Int
    ): ByteArray

    companion object {
        private const val TAG = "ZETIC MLange Yolov8 Wrapper"

        // Used to load the 'yolov8_jni_wrapper' library on application startup.
        init {
            System.loadLibrary("yolov8_jni_wrapper")
        }
    }
}
