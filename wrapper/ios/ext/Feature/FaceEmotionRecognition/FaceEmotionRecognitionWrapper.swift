import Foundation
import UIKit

public class FaceEmotionRecognitionWrapper {
    let wrapperPtr: Int64
    public init() {
        wrapperPtr = FaceEmotionRecognitionWrapper.nativeInit()
    }
    
    deinit {
        FaceEmotionRecognitionWrapper.nativeDeinit(wrapperPtr)
    }
    
    public func preprocess(_ image: UIImage, _ roi: Box) -> Data {
        var count: Int32 = 0
        let cgImage = FeatureUtils.createCGImage(from: image)!
        let floatPointer = FaceEmotionRecognitionWrapper.nativePreprocess(wrapperPtr, cgImage, roi.xmin, roi.ymin, roi.xmax, roi.ymax, &count)
        return Data(bytes: floatPointer, count: Int(count) * MemoryLayout<Float>.size)
    }
    
    public func postprocess(_ outputData: inout [Data]) -> FaceEmotionRecognitionResult {
        var pointers = [ FeatureUtils.dataToMutableFloatPointer(data: &outputData[0])! ]
        
        var emotion = [CChar](repeating: 0, count: 32)
        var confidence: Float = 0
        pointers.withUnsafeMutableBufferPointer { buffer in
            FaceEmotionRecognitionWrapper.nativePostprocess(wrapperPtr, buffer.baseAddress!, &emotion, &confidence)
        }
        
        return FaceEmotionRecognitionResult(emotion: String(cString: emotion), confidence: confidence)
    }
    
    @_silgen_name("nativeInitFaceEmotionRecognition")
    static func nativeInit() -> Int64
    
    @_silgen_name("nativeDeinitFaceEmotionRecognition")
    static func nativeDeinit(_ zeticMLangeFeatureFaceEmotionRecognitionPtr: Int64)
    
    @_silgen_name("nativePreprocessFaceEmotionRecognition")
    static func nativePreprocess(_ zeticMLangeFeatureFaceEmotionRecognitionPtr: Int64, _ cgImage: CGImage, _ xmin: Float, _ ymin: Float, _ xmax: Float, _ ymax: Float, _ countPtr: UnsafeMutablePointer<Int32>) -> UnsafeMutablePointer<Float>
    
    @_silgen_name("nativePostprocessFaceEmotionRecognition")
    static func nativePostprocess(_ zeticMLangeFeatureFaceEmotionRecognitionPtr: Int64, _ outputData: UnsafeMutablePointer<UnsafeMutablePointer<Float>>?, _ emotion: UnsafeMutablePointer<Int8>, _ confidence: UnsafeMutablePointer<Float>)
}

