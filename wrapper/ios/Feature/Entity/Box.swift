public struct Box {
    public let xmin: Float
    public let ymin: Float
    public let xmax: Float
    public let ymax: Float
    
    public init(xmin: Float, ymin: Float, xmax: Float, ymax: Float) {
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
    }
}
