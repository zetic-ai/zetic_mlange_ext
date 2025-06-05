#pragma once

#include "feature_opencv.h"

#import <CoreGraphics/CoreGraphics.h>

extern "C" {

cv::Mat _getCVMatFromCGImageRef(CGImageRef cgImage);
CGImageRef _cvMatToCGImage_wo_arg();
cv::Mat nativeConvertToBGR(const uint8_t* data, int width, int height, int formatCode);

}
