import SwiftUI
import Combine
import AVFoundation

public struct CameraPreview: UIViewRepresentable {
    let previewLayer: AVCaptureVideoPreviewLayer
    
    public init(previewLayer: AVCaptureVideoPreviewLayer) {
        self.previewLayer = previewLayer
    }
    
    public func makeUIView(context: Context) -> UIView {
        let view = UIView()
        view.layer.addSublayer(previewLayer)
        return view
    }
    
    public func updateUIView(_ uiView: UIView, context: Context) {
        DispatchQueue.main.async {
            previewLayer.frame = uiView.bounds
        }
    }
}

public struct CameraPreviewView: View {
    let previewLayer: AVCaptureVideoPreviewLayer
    
    public init(previewLayer: AVCaptureVideoPreviewLayer) {
        self.previewLayer = previewLayer
    }
    
    public var body: some View {
        GeometryReader { geometry in
            let aspectRatio: CGFloat = geometry.size.height / geometry.size.width
            let targetSize = CGSize(width: geometry.size.width, height: geometry.size.width * aspectRatio)
            CameraPreview(previewLayer: previewLayer)
                .customFrame(targetWidth: targetSize.width, targetHeight: targetSize.height,
                             geometryWidth: geometry.size.width, geometryHeight: geometry.size.height)
                .ignoresSafeArea()
        }
    }
}
