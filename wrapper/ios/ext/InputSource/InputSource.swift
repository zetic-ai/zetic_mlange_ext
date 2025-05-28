import Foundation
import SwiftUI

public protocol InputSource<Input> {
    associatedtype Input
    func acquire(_ frame: @escaping (Input) -> Void)
    func stop()
}
