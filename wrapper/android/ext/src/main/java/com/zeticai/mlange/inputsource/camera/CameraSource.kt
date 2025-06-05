package com.zeticai.mlange.inputsource.camera

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.ImageFormat
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.media.Image
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.util.Size
import android.view.SurfaceHolder
import com.zeticai.mlange.inputsource.InputSource

@SuppressLint("MissingPermission")
open class CameraSource @JvmOverloads constructor(
    context: Context,
    private val preview: SurfaceHolder,
    preferredSize: Size,
    cameraDirection: CameraDirection = CameraDirection.BACK
) : InputSource {
    private val manager: CameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    private val cameraId: String = manager.cameraIdList[cameraDirection.id]
    private val handler = Handler(
        HandlerThread("camera2").apply {
            start()
        }.looper
    )

    private val imageReader = ImageReader.newInstance(
        preferredSize.width, preferredSize.height, ImageFormat.JPEG, 2
    )

    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private val cameraDeviceStateCallback = object : CameraDevice.StateCallback() {
        override fun onOpened(camera: CameraDevice) {
            cameraDevice = camera
            cameraDevice?.createCaptureSession(
                listOf(
                    preview.surface,
                    imageReader.surface,
                ), cameraCaptureSessionStateCallback, handler
            )
        }

        override fun onDisconnected(camera: CameraDevice) {
            cameraDevice?.close()
        }

        override fun onError(camera: CameraDevice, error: Int) {

        }
    }

    private val previewBuilder by lazy {
        cameraDevice!!.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
            addTarget(imageReader.surface)
            addTarget(preview.surface)
        }
    }

    private val cameraCaptureSessionStateCallback = object : CameraCaptureSession.StateCallback() {
        override fun onConfigured(session: CameraCaptureSession) {
            captureSession = session
            session.setRepeatingRequest(previewBuilder.build(), null, handler)
        }

        override fun onConfigureFailed(session: CameraCaptureSession) {

        }
    }
    private val previewSurfaceHolderCallback = object : SurfaceHolder.Callback {
        override fun surfaceCreated(holder: SurfaceHolder) {
        }

        override fun surfaceChanged(
            holder: SurfaceHolder, format: Int, width: Int, height: Int
        ) {
            manager.openCamera(cameraId, cameraDeviceStateCallback, handler)
        }

        override fun surfaceDestroyed(holder: SurfaceHolder) {
            cameraDevice?.close()
        }
    }

    private fun processCameraImage(image: Image): ByteArray {
        val buffer = image.planes[0].buffer
        val array = ByteArray(buffer.remaining())
        buffer.get(array)
        return array
    }

    fun close() {
        cameraDevice?.close()
        captureSession?.close()
        imageReader.close()
    }

    override fun acquire(frame: (ByteArray) -> Unit) {
        imageReader.setOnImageAvailableListener(
            {
                val image = it.acquireLatestImage() ?: return@setOnImageAvailableListener
                frame(processCameraImage(image))
                image.close()
            }, handler
        )
        preview.addCallback(previewSurfaceHolderCallback)
    }
}