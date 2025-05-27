#include "yolov8_swift_wrapper.hpp"
#include <chrono>


extern "C" {

using namespace std::chrono;

// TODO: Currently use as static to reduce conversion time from bitmap to cv::mat
static int8_t* blob;
static DLResultC* results;

long nativeInitDetect(const char* coco_file_path) {
    ZeticMLangeYoloV8Feature* yolo_v8_feature = new ZeticMLangeYoloV8Feature(YOLO_DETECT_V8, coco_file_path);
    return reinterpret_cast<long>(yolo_v8_feature);
}

long nativeInitClassifier(const char* coco_file_path) {
    ZeticMLangeYoloV8Feature* yolo_v8_feature = new ZeticMLangeYoloV8Feature(YOLO_CLS, coco_file_path);
    return reinterpret_cast<long>(yolo_v8_feature);
}

void nativeDeinitFeature(long yolo_v8_feature_ptr) {
    delete reinterpret_cast<ZeticMLangeYoloV8Feature*>(yolo_v8_feature_ptr);
}


// TODO: User should free buffer
int8_t* nativeFeaturePreprocess(long yolo_v8_feature_ptr, void *base_address, int width, int height, int bytes_per_row, int* count_ptr) {
    cv::Mat img(height, width, CV_8UC4, base_address, bytes_per_row);
    cv::cvtColor(img, img, cv::COLOR_BGRA2BGR);
    return nativePreprocess(yolo_v8_feature_ptr, img, count_ptr);
}

DLResultC* nativeFeaturePostprocess(long yolo_v8_feature_ptr, int8_t* output, int* count) {
    ZeticMLangeYoloV8Feature* yolo_v8_feature = reinterpret_cast<ZeticMLangeYoloV8Feature*>(yolo_v8_feature_ptr);
    
    Zetic_MLange_Feature_Result_t ret = ZETIC_MLANGE_FEATURE_FAIL;
    std::vector<DL_RESULT> res;
    
    ret = yolo_v8_feature->postprocess(res, (void*)output);
    
    if (ret != ZETIC_MLANGE_FEATURE_SUCCESS) {
        printf("Failed to postprocess!");
        return {};
    }
    
    *count = std::min(static_cast<int>(res.size()), MAX_YOLO_RESULT_COUNT);
    if (!results) {
        results = (DLResultC*)malloc(sizeof(DLResultC) * MAX_YOLO_RESULT_COUNT);
    }
    
    for (int i = 0; i < *count; i++) {
        results[i].class_id = res[i].class_id;
        results[i].confidence = res[i].confidence;
        results[i].x = res[i].box.x;
        results[i].y = res[i].box.y;
        results[i].width = res[i].box.width;
        results[i].height = res[i].box.height;
    }
    
    return results;
}


int8_t* nativeFeaturePreprocessWithFrame(long yolo_v8_feature_ptr, int8_t* frame, int width, int height, int formatCode, int* count_ptr) {
    cv::Mat bgrMat = MLangeFeatureOpenCV::convertToBGR(reinterpret_cast<const uint8_t*>(frame), width, height, formatCode);
    
    return nativePreprocess(yolo_v8_feature_ptr, bgrMat, count_ptr);
}

int8_t* nativePreprocess(long yolo_v8_feature_ptr, cv::Mat& mat, int* count_ptr) {
    ZeticMLangeYoloV8Feature* yolo_v8_feature = reinterpret_cast<ZeticMLangeYoloV8Feature*>(yolo_v8_feature_ptr);
    
    cv::Mat output_mat;
    yolo_v8_feature->preprocess(mat, output_mat);
    
    size_t len_blob = output_mat.total() * output_mat.channels() * sizeof(float);
    *count_ptr = len_blob;
    if (!blob) {
        blob = new int8_t[len_blob];
    }
    
    auto ret = yolo_v8_feature->getByteArrayFromImage(output_mat, blob);
    if (ret != ZETIC_MLANGE_FEATURE_SUCCESS) {
        printf("Failed to get blob data from pre-processed image for ZeticMLangeYolov8");
        return nullptr;
    }
    return blob;
}

}   // extern "C"
