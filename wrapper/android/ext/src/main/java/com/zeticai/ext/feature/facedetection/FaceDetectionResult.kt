package com.zeticai.ext.feature.facedetection

import com.zeticai.ext.feature.entity.Box

data class FaceDetectionResult(
    var bbox: Box,
    var score: Float
) {
    fun reset(bbox: Box, score: Float) {
        this.bbox = bbox
        this.score = score
    }
}
