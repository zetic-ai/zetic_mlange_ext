package com.zeticai.ext.feature.objectdetection

import android.content.Context
import com.zeticai.ext.feature.AutoSelectPipelineFeature
import com.zeticai.ext.feature.PipelineFeature
import com.zeticai.ext.feature.objectdetection.yolov8.YOLOv8

class ObjectDetection(
    private val context: Context
) : AutoSelectPipelineFeature<ObjectDetectionResult>() {

    override fun selectFeature(): PipelineFeature<ObjectDetectionResult> {
        return YOLOv8(context)
    }
}