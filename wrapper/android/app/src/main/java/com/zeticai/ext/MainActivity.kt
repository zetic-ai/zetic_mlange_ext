package com.zeticai.ext

import android.content.Context
import android.graphics.ImageFormat
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.os.Bundle
import android.util.Size
import android.widget.FrameLayout
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.zeticai.mlange.feature.objectdetection.ObjectDetection
import com.zeticai.mlange.feature.objectdetection.yolov8.YOLOResult
import com.zeticai.mlange.inputsource.camera.CameraSource
import com.zeticai.mlange.inputsource.camera.PreviewSurfaceView
import com.zeticai.mlange.inputsource.camera.YOLOResultSurfaceView
import com.zeticai.mlange.pipeline.ZeticMLangePipeline

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.root)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        val surface = PreviewSurfaceView(this)
        val yolo = YOLOResultSurfaceView(this)

        findViewById<FrameLayout>(R.id.root).addView(surface)
        findViewById<FrameLayout>(R.id.root).addView(yolo)
        val preferredSize = getSizeForMinResolution(this, 640)

        surface.updateSizeKeepRatio(preferredSize)
        yolo.updateSizeKeepRatio(preferredSize)

        val objectDetectionPipeline = ZeticMLangePipeline(
            feature = ObjectDetection(this),
            inputSource = CameraSource(this, surface.holder, preferredSize),
        )

        objectDetectionPipeline.loop { runOnUiThread {
            yolo.visualize(YOLOResult(it.value), preferredSize, true)
        }}

    }
    private fun getSizeForMinResolution(context: Context, minDimension: Int): Size {
        val manager: CameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val cameraId: String = manager.cameraIdList[0]
        val characteristics: CameraCharacteristics = manager.getCameraCharacteristics(cameraId)

        val sizes = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
            ?.getOutputSizes(ImageFormat.JPEG)

        if (sizes.isNullOrEmpty()) {
            throw Exception("No camera found")
        }

        for (i in sizes.size - 1 downTo 0) {
            val size = sizes[i]
            if (size.width >= minDimension && size.height >= minDimension) {
                return size
            }
        }

        throw Exception("No size found")
    }

}