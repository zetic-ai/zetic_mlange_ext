package com.zeticai.mlange.feature.objectdetection.yolov8

import android.content.Context
import com.zeticai.mlange.core.common.DataUtils
import com.zeticai.mlange.core.tensor.Tensor
import com.zeticai.mlange.feature.PipelineFeature
import com.zeticai.mlange.feature.PipelineInferenceModel
import com.zeticai.mlange.feature.ZeticMLangeModelWrapper
import com.zeticai.mlange.feature.objectdetection.ObjectDetectionResult
import com.zeticai.mlange.feature.vision.OpenCVImageUtilsWrapper
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer

class YOLOv8(
    private val context: Context,
    cocoYamlFilePath: String = findCocoYamlFilePath(context),
    private val model: PipelineInferenceModel = ZeticMLangeModelWrapper(
        context,
        "ztp_97aT0F0HtHQ5Q3dasRCIAoxKH0O0YKJUyvOB",
        "b9f5d74e6f644288a32c50174ded828e"
    ),
) : PipelineFeature<ObjectDetectionResult> {
    val wrapper = YOLOv8Wrapper(cocoYamlFilePath)
    private val openCVImageUtilsWrapper = OpenCVImageUtilsWrapper()

    override fun preprocess(input: Array<ByteArray>): Array<ByteArray> {
        val imagePtr = openCVImageUtilsWrapper.frame(input[0], 90)
        return arrayOf(wrapper.preprocess(imagePtr))
    }

    override fun postprocess(output: Array<ByteBuffer>): ObjectDetectionResult {
        return ObjectDetectionResult(wrapper.postprocess(output[0]).value)
    }

    override fun run(input: Array<ByteArray>): ObjectDetectionResult {
        val pre = preprocess(input)
        val output = model.inference(pre.map {
            Tensor.of(it)
        }.toTypedArray())
        return postprocess(output.map {
            it.data
        }.toTypedArray())
    }

    companion object {
        private fun findCocoYamlFilePath(context: Context): String {
            val cocoYamlSamplePath = "coco.yaml"
            copyFileFromAssetsToData(context, cocoYamlSamplePath)
            val cocoYamlFile = File(context.filesDir, cocoYamlSamplePath)
            return cocoYamlFile.absolutePath
        }

        private fun copyFileFromAssetsToData(context: Context, fileName: String) {
            val assetManager = context.assets
            val file = assetManager.open(fileName)
            val outFile = File(context.filesDir, fileName)
            val out = FileOutputStream(outFile)

            val buffer = ByteArray(1024)
            var read: Int
            while ((file.read(buffer).also { read = it }) != -1) {
                out.write(buffer, 0, read)
            }

            file.close()
            out.flush()
            out.close()
        }
    }
}
