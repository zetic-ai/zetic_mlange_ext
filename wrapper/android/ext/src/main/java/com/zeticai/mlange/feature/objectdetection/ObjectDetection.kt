package com.zeticai.mlange.feature.objectdetection

import android.content.Context
import com.zeticai.mlange.feature.AutoSelectPipelineFeature
import com.zeticai.mlange.feature.PipelineFeature
import com.zeticai.mlange.feature.objectdetection.yolov8.YOLOv8

class ObjectDetection(
    private val context: Context
) : AutoSelectPipelineFeature<ObjectDetectionResult>() {

    override fun selectFeature(): PipelineFeature<ObjectDetectionResult> {
        return YOLOv8(context)
    }
}