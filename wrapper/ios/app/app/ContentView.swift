import SwiftUI
import AVFoundation
import Vision

import ZeticMLange
import ext

struct ContentView: View {
    @StateObject private var cameraSource: CameraSource
    @StateObject private var pipeline: ZeticMLangePipeline<CameraFrame, ObjectDetectionResult>
    
    init() {
        let camera = CameraSource()
        let detection = ObjectDetection()
        _pipeline = StateObject(wrappedValue: ZeticMLangePipeline(feature: detection, inputSource: camera))
        _cameraSource = StateObject(wrappedValue: camera)
    }
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                if let previewLayer = cameraSource.previewLayer {
                    CameraPreviewView(previewLayer: previewLayer)
                }
                
                if let detections = pipeline.latestResult {
                    DetectionsView(detectionResult: detections, cameraResolution: cameraSource.resolution)
                }
            }
        }
        .onAppear {
            pipeline.startLoop()
        }
        .onDisappear {
            pipeline.stopLoop()
        }
    }
}
