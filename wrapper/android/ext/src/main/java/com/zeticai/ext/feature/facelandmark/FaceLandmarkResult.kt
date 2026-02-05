package com.zeticai.ext.feature.facelandmark

import com.zeticai.ext.feature.entity.Landmark

data class FaceLandmarkResult(
    val landmarks: List<Landmark>,
    val confidence: Float
)
