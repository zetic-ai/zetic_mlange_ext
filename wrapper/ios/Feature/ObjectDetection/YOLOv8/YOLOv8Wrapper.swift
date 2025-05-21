import Foundation
import Darwin
import SwiftUI
import CoreGraphics

public class YOLOv8Wrapper {
    let wrapperPtr: Int64
    var data: Data? = nil
    
    public init(_ cocoFilePath: String) {
        let cocoFilePath = cocoFilePath.replacingOccurrences(of: "file://", with: "")
        let cStrCocoFilePath = strdup(cocoFilePath)!
        
        // Currently support detector only
        wrapperPtr = YOLOv8Wrapper.nativeInitDetect(cStrCocoFilePath)
    }
    
    deinit {
        YOLOv8Wrapper.nativeDeinitFeature(wrapperPtr)
    }
    
    public func featurePreprocess(_ baseAddress: UnsafeMutableRawPointer, _ width: Int32, _ height: Int32, _ bytesPerRow: Int32) -> Data {
        var count: Int32 = 0
        let bytePointer = YOLOv8Wrapper.nativeFeaturePreprocess(wrapperPtr, baseAddress, width, height, bytesPerRow, &count)
        
        return convertBytePointerToDataWithCaching(bytePointer, count)
    }
    
    public func featurePostprocess(_ outputFloatArray: UnsafeMutablePointer<UInt8>) -> [YOLOv8Result] {
        var count: Int32 = 0
        let resultsPtr = YOLOv8Wrapper.nativeFeaturePostprocess(wrapperPtr, outputFloatArray, &count)
        
        var results: [YOLOv8Result] = []
        for i in 0..<Int(count) {
            let result = resultsPtr[i]
            
            let box = [result.x, result.y, result.width, result.height]
            
            results.append(YOLOv8Result(
                classId: result.classId,
                confidence: result.confidence,
                box: box
            ))
        }
        return results
    }
    
    public func featurePreprocessWithFrame(_ frameData: UnsafePointer<UInt8>, _ width: Int, _ height: Int, _ formatCode: Int) -> Data {
        var count: Int32 = 0
        let bytePointer = YOLOv8Wrapper.nativeFeaturePreprocessWithFrame(wrapperPtr, frameData, Int32(width), Int32(height), Int32(formatCode), &count)
        return convertBytePointerToDataWithCaching(bytePointer, count)
    }
    
    @_silgen_name("nativeInitDetect")
    static func nativeInitDetect(_ cocoFilePath: UnsafeMutablePointer<Int8>) -> Int64
    
    @_silgen_name("nativeInitClassifier")
    static func nativeInitClassifier(_ cocoFilePath: UnsafeMutablePointer<Int8>) -> Int64
    
    @_silgen_name("nativeDeinitFeature")
    static func nativeDeinitFeature(_ modelPtr: Int64)
    
    @_silgen_name("nativeFeaturePreprocess")
    static func nativeFeaturePreprocess(_ modelPtr: Int64, _ baseAddress: UnsafeMutableRawPointer, _ width: Int32, _ height: Int32, _ bytesPerRow: Int32, _ countPtr: UnsafeMutablePointer<Int32>) -> UnsafeMutablePointer<UInt8>
    
    @_silgen_name("nativeFeaturePostprocess")
    static func nativeFeaturePostprocess(_ modelPtr: Int64, _ outputFloatArray: UnsafeMutablePointer<UInt8>, _ countPtr: UnsafeMutablePointer<Int32>) -> UnsafeMutablePointer<DLResultC>
    
    @_silgen_name("nativeFeaturePreprocessWithFrame")
    static func nativeFeaturePreprocessWithFrame(_ modelPtr: Int64, _ frameData: UnsafePointer<UInt8>, _ width: Int32, _ height: Int32, _ formatCode: Int32, _ countPtr: UnsafeMutablePointer<Int32>)-> UnsafeMutablePointer<UInt8>
    
    private func convertBytePointerToDataWithCaching(_ ptr: UnsafeMutablePointer<UInt8>, _ count: Int32) -> Data {
        let floatCount = Int(count) / MemoryLayout<Float>.size
        let floatPointer = UnsafeRawPointer(ptr).bindMemory(to: Float.self, capacity: floatCount)
        
        if data == nil {
            data = Data(bytes: floatPointer, count: Int(count))
        } else {
            data?.removeAll(keepingCapacity: true)
            data?.count = Int(count)
            data?.withUnsafeMutableBytes { destPointer in
                guard let destBaseAddress = destPointer.baseAddress else {
                    return
                }
                memcpy(destBaseAddress, floatPointer, Int(count))
            }
        }
        return data!
    }
    
    struct DLResultC {
        let classId: Int32
        let confidence: Float
        let x: Int32
        let y: Int32
        let width: Int32
        let height: Int32
    }
    
}
