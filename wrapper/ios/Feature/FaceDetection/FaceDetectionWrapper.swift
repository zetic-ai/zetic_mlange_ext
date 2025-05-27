import Foundation
import UIKit
import CoreGraphics

public class FaceDetectionWrapper {
    let wrapperPtr: Int64
    public init() {
        wrapperPtr = FaceDetectionWrapper.nativeInit()
    }
    
    deinit {
        FaceDetectionWrapper.nativeDeinit(wrapperPtr)
    }
    
    public func preprocess(_ image: UIImage) -> Data {
        var count: Int32 = 0
        let cgImage = FeatureUtils.createCGImage(from: image)!
        let floatPointer = FaceDetectionWrapper.nativePreprocess(wrapperPtr, cgImage, &count)
        return Data(bytes: floatPointer, count: Int(count) * MemoryLayout<Float>.size)
    }
    
    public func postprocess(_ outputData: inout [Data]) -> Array<FaceDetectionResult> {
        outputData.sort(by: {
            return $0.count > $1.count
        })
        var pointers = [ FeatureUtils.dataToMutableFloatPointer(data: &outputData[0])!,
                         FeatureUtils.dataToMutableFloatPointer(data: &outputData[1])! ]
        
        
        var outputSize: Int32 = 0
        let resultPointers = pointers.withUnsafeMutableBufferPointer { buffer in
            FaceDetectionWrapper.nativePostprocess(wrapperPtr, buffer.baseAddress!, &outputSize)
        }
        var resultArray: [FaceDetectionResult] = []
        
        resultPointers.withMemoryRebound(to: FaceDetectionResult.self, capacity: Int(outputSize)) { resultPointer in
            for i in 0..<Int(outputSize) {
                let element = resultPointer[i]
                let faceResult = FaceDetectionResult(bbox: Box(xmin: element.bbox.xmin,
                                                               ymin: element.bbox.ymin,
                                                               xmax: element.bbox.xmax,
                                                               ymax: element.bbox.ymax),
                                                     confidence: element.confidence)
                resultArray.append(faceResult)
            }
        }
        
        return resultArray
    }
    
    @_silgen_name("nativeInitFaceDetection")
    static func nativeInit() -> Int64
    
    @_silgen_name("nativeDeinitFaceDetection")
    static func nativeDeinit(_ zeticMLangeFeatureFaceDetectionPtr: Int64)
    
    @_silgen_name("nativePreprocessFaceDetection")
    static func nativePreprocess(_ zeticMLangeFeatureFaceDetectionPtr: Int64, _ image: CGImage, _ countPtr: UnsafeMutablePointer<Int32>) -> UnsafeMutablePointer<Float>
    
    @_silgen_name("nativePostprocessFaceDetection")
    static func nativePostprocess(_ zeticMLangeFeatureFaceDetectionPtr: Int64, _ outputData: UnsafeMutablePointer<UnsafeMutablePointer<Float>>?, _ outputSize: UnsafeMutablePointer<Int32>) -> UnsafePointer<FaceDetectionResult>
}
