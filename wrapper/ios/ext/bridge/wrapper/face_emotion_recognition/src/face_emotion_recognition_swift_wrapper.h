
#pragma once

#include "face_emotion_recognition_feature.h"
#include "zetic_feature_types.h"
#include "opencv_swift_wrapper.h"

#import <CoreGraphics/CoreGraphics.h>

using namespace ZeticMLange;

extern "C" {
long nativeInitFaceEmotionRecognition();
void nativeDeinitFaceEmotionRecognition(long face_emotion_recognition_feature_ptr);

float* nativePreprocessFaceEmotionRecognition(long face_emotion_recognition_feature_ptr, CGImageRef input_image, float x_min, float y_min, float x_max, float y_max, int* blob_size);

void nativePostprocessFaceEmotionRecognition(long face_emotion_recognition_feature_ptr, float** output_data, char* emotion, float* confidence);
}
