#pragma once

#include "face_landmark_feature.h"
#include "zetic_feature_types.h"
#include "face_landmark_result.h"
#include "opencv_swift_wrapper.h"

#include <vector>

#import <CoreGraphics/CoreGraphics.h>

using namespace ZeticMLange;

extern "C" {
long nativeInitFaceLandmark();
void nativeDeinitFaceLandmark(long face_landmark_feature_ptr);

float* nativePreprocessFaceLandmark(long face_landmark_feature_ptr, CGImageRef input_image, float x_min, float y_min, float x_max, float y_max, int* blob_size);

Landmark* nativePostprocessFaceLandmark(long face_landmark_feature_ptr, float** output_data, float* confidence, int* output_size);
}
