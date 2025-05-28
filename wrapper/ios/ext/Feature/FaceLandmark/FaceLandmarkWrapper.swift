import Foundation
import UIKit

public class FaceLandmarkWrapper {
    let wrapperPtr: Int64
    
    public init() {
        wrapperPtr = FaceLandmarkWrapper.nativeInit()
    }
    
    deinit {
        FaceLandmarkWrapper.nativeDeinit(wrapperPtr)
    }
    
    public func preprocess(_ image: UIImage, _ roi: Box) -> Data {
        var count: Int32 = 0
        let cgImage = FeatureUtils.createCGImage(from: image)!
        let floatPointer = FaceLandmarkWrapper.nativePreprocess(wrapperPtr, cgImage, roi.xmin, roi.ymin, roi.xmax, roi.ymax, &count)
        return Data(bytes: floatPointer, count: Int(count) * MemoryLayout<Float>.size)
    }
    
    public func postprocess(_ outputData: inout [Data]) -> FaceLandmarkResult {
        outputData.sort(by: {
            return $0.count < $1.count
        })
        var pointers = [ FeatureUtils.dataToMutableFloatPointer(data: &outputData[0])!,
                         FeatureUtils.dataToMutableFloatPointer(data: &outputData[1])! ]
        
        var confidence: Float = 0
        var outputSize: Int = 0
        let result = pointers.withUnsafeMutableBufferPointer { buffer in
            FaceLandmarkWrapper.nativePostprocess(wrapperPtr, buffer.baseAddress!, &confidence, &outputSize)
        }
        
        let landmarks = result.withMemoryRebound(to: Landmark.self, capacity: outputSize) { resultPointer in
            return (0..<outputSize).map { i in
                return Landmark(x: resultPointer[i].x, y: resultPointer[i].y, z: resultPointer[i].z)
            }
        }
        return FaceLandmarkResult(faceLandmark: landmarks, confidence: confidence);
    }
    
    @_silgen_name("nativeInitFaceLandmark")
    static func nativeInit() -> Int64
    
    @_silgen_name("nativeDeinitFaceLandmark")
    static func nativeDeinit(_ wrapperPtr: Int64)
    
    @_silgen_name("nativePreprocessFaceLandmark")
    static func nativePreprocess(_ wrapperPtr: Int64, _ image: CGImage, _ xmin: Float, _ ymin: Float, _ xmax: Float, _ ymax: Float, _ countPtr: UnsafeMutablePointer<Int32>) -> UnsafeMutablePointer<Float>
    
    @_silgen_name("nativePostprocessFaceLandmark")
    static func nativePostprocess(_ wrapperPtr: Int64, _ outputData: UnsafeMutablePointer<UnsafeMutablePointer<Float>>?, _ confidencePtr: UnsafeMutablePointer<Float>, _ outputSizePtr: UnsafeMutablePointer<Int>) -> UnsafeMutablePointer<Landmark>
}

