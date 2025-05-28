import Foundation

public class WhisperWrapper {
    public init(_ vocabularyPath: String) {
        let path = vocabularyPath.replacingOccurrences(of: "file://", with: "")
        let cStrPath = strdup(path)!
        WhisperWrapper.nativeInit(cStrPath)
    }
    
    deinit {
        WhisperWrapper.nativeDeinit()
    }
    
    public func process(_ audio: [Float]) -> [Float] {
        var count: Int32 = 0
        
        let pointer = UnsafeMutablePointer<Float>.allocate(capacity: audio.count)
        pointer.initialize(from: audio, count: audio.count)
        
        let processResult = WhisperWrapper.nativeProcess(pointer, audio.count, &count)
        return safePointerToArray(pointer: processResult, count: Int(count))
    }
    
    public func decodeToken(_ ids: [Int32], _ skipSpecialTokens: Bool) -> String {
        var count: Int32 = 0
        
        let pointer = UnsafeMutablePointer<Int32>.allocate(capacity: ids.count)
        pointer.initialize(from: ids, count: ids.count)
        
        let cString = WhisperWrapper.nativeDecodeToken(pointer, ids.count, skipSpecialTokens, &count)
        return String(cString: cString)
    }
    
    @_silgen_name("nativeInitWhisper")
    static func nativeInit(_ vocabularyPath: UnsafeMutablePointer<Int8>)
    
    @_silgen_name("nativeDeinitWhisper")
    static func nativeDeinit()
    
    @_silgen_name("nativeProcessWhisper")
    static func nativeProcess(_ audio: UnsafeMutablePointer<Float>, _ audioSize: Int, _ returnSize: UnsafeMutablePointer<Int32>) -> UnsafeMutablePointer<Float>
    
    @_silgen_name("nativeDecodeTokenWhisper")
    static func nativeDecodeToken(_ ids: UnsafeMutablePointer<Int32>, _ idsSize: Int, _ skipSpecialTokens: Bool, _ returnSize: UnsafeMutablePointer<Int32>) -> UnsafeMutablePointer<Int8>
    
    func safePointerToArray(pointer: UnsafeMutablePointer<Float>, count: Int) -> [Float] {
        return pointer.withMemoryRebound(to: Float.self, capacity: count) { pointer in
            return Array(UnsafeBufferPointer(start: pointer, count: count))
        }
    }
}
