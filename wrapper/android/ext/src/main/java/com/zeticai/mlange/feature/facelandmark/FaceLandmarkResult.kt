package com.zeticai.mlange.feature.facelandmark

import com.zeticai.mlange.feature.entity.Landmark

data class FaceLandmarkResult(
    val landmarks: List<Landmark>,
    val confidence: Float
)
