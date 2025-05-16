package com.zeticai.mlange.feature.entity

data class Box(
    var xMin: Float,
    var yMin: Float,
    var xMax: Float,
    var yMax: Float
) {
    fun reset(xMin: Float, yMin: Float, xMax: Float, yMax: Float) {
        this.xMin = xMin
        this.yMin = yMin
        this.xMax = xMax
        this.yMax = yMax
    }
}
