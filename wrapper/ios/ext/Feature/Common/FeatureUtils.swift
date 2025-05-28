import Foundation
import CoreGraphics
import UIKit

public class FeatureUtils {
    static func createCGImage(from image: UIImage) -> CGImage? {
        if let cgImage = image.cgImage {
            return cgImage
        } else if let ciImage = image.ciImage {
            // Render the CIImage into a CGImage
            let context = CIContext()
            return context.createCGImage(ciImage, from: ciImage.extent)
        }
        return nil
    }
    
    
    public static func dataToMutableFloatPointer(data: inout Data) -> UnsafeMutablePointer<Float>? {
        // Check that the data size is a multiple of the size of Float
        guard data.count % MemoryLayout<Float>.size == 0 else {
            print("Data size is not a multiple of Float size.")
            return nil
        }
        
        // Convert Data to UnsafeMutablePointer<Float>
        let floatPointer = data.withUnsafeMutableBytes { (rawBufferPointer) -> UnsafeMutablePointer<Float>? in
            // Check if the base address is not nil
            guard let baseAddress = rawBufferPointer.baseAddress else {
                return nil
            }
            
            // Return the pointer as UnsafeMutablePointer<Float>
            return baseAddress.assumingMemoryBound(to: Float.self)
        }
        
        return floatPointer
    }
    
    public static func dataToMutableBytePointer(data: inout Data) -> UnsafeMutablePointer<UInt8>? {
        // Check that the data size is a multiple of the size of Float
        guard data.count % MemoryLayout<UInt8>.size == 0 else {
            print("Data size is not a multiple of Float size.")
            return nil
        }
        
        // Convert Data to UnsafeMutablePointer<Float>
        let floatPointer = data.withUnsafeMutableBytes { (rawBufferPointer) -> UnsafeMutablePointer<UInt8>? in
            // Check if the base address is not nil
            guard let baseAddress = rawBufferPointer.baseAddress else {
                return nil
            }
            
            // Return the pointer as UnsafeMutablePointer<Float>
            return baseAddress.assumingMemoryBound(to: UInt8.self)
        }
        
        return floatPointer
    }
}
