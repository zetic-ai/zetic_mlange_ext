#ifndef zetic_mlange_feature_yolov8_hpp
#define zetic_mlange_feature_yolov8_hpp

#include "getopt.h"
#include "yolov8_feature.h"
#include "zetic_feature_types.h"
#include "feature_opencv.h"

#import <CoreGraphics/CoreGraphics.h>

//#ifdef __cplusplus
extern "C" {
//#endif

long nativeInitDetect(const char* coco_file_path);
long nativeInitClassifier(const char* coco_file_path);
void nativeDeinitFeature(long yolo_v8_feature_ptr);

int8_t* nativePreprocess(long yolo_v8_feature_ptr, cv::Mat& mat, int* count_ptr);
int8_t* nativeFeaturePreprocess(long yolo_v8_feature_ptr, void *base_address, int width, int height, int bytes_per_row, int* count_ptr);
int8_t* nativeFeaturePreprocessWithFrame(long yolo_v8_feature_ptr, int8_t* frame, int width, int height, int formatCode, int* count_ptr);

DLResultC* nativeFeaturePostprocess(long yolo_v8_feature_ptr, int8_t* output_float_array, int* count_ptr);

//#ifdef __cplusplus
}   // extern "C"
//#endif

#endif /* zetic_mlange_feature_yolov8_hpp */
