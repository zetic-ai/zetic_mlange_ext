#pragma once

#include "face_detection_feature.h"
#include "zetic_feature_types.h"
#include "face_detection_result.h"
#include "opencv_swift_wrapper.h"

#import <CoreGraphics/CoreGraphics.h>

using namespace ZeticMLange;

extern "C" {
long nativeInitFaceDetection();
void nativeDeinitFaceDetection(long face_detection_feature_ptr);

float* nativePreprocessFaceDetection(long face_detection_feature_ptr, CGImageRef input_image, int* count_ptr);
FaceDetectionResult* nativePostprocessFaceDetection(long face_detection_feature_ptr, float** output_data, int* output_size);
}
