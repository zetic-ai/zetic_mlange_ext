package com.zeticai.mlange.feature.objectdetection.yolov8

import com.zeticai.mlange.feature.entity.Box

data class YOLOObject (
    var classId: Int,
    var confidence: Float,
    var box: Box
) {
    fun reset(classId: Int, confidence: Float, box: Box) {
        this.classId = classId
        this.confidence = confidence
        this.box = box
    }
}