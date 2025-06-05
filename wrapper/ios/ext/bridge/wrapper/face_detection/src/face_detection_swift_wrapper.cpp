#include "face_detection_swift_wrapper.h"

extern "C" {

static float* blob;
static std::vector<FaceDetectionResult> results;

long nativeInitFaceDetection() {
    FaceDetectionFeature* face_detection_feature = new FaceDetectionFeature();
    return reinterpret_cast<long>(face_detection_feature);
}

void nativeDeinitFaceDetection(long face_detection_feature_ptr) {
    delete reinterpret_cast<FaceDetectionFeature*>(face_detection_feature_ptr);
}

float* nativePreprocessFaceDetection(long face_detection_feature_ptr, CGImageRef input_image, int* count_ptr) {
    FaceDetectionFeature* face_detection_feature = reinterpret_cast<FaceDetectionFeature*>(face_detection_feature_ptr);
    cv::Mat img = _getCVMatFromCGImageRef(input_image);
    
    if (img.empty())
        return nullptr;
    
    cv::Mat input_data;
    auto ret = face_detection_feature->preprocess(img, input_data);
    if (ret != ZETIC_MLANGE_FEATURE_SUCCESS)
        return nullptr;
    
    int len_blob = input_data.total() * 3;
    *count_ptr = len_blob;
    if (!blob) {
        blob = new float[len_blob];
    }
    
    MLangeFeatureOpenCV mlange_feature_opencv;
    ret = mlange_feature_opencv.getFlatFloatArrayFromImage(input_data, blob);
    if (ret != ZETIC_MLANGE_FEATURE_SUCCESS)
        return nullptr;
    
    return blob;
}

FaceDetectionResult* nativePostprocessFaceDetection(long face_detection_feature_ptr, float** output_data, int* output_size) {
    FaceDetectionFeature* face_detection_feature = reinterpret_cast<FaceDetectionFeature*>(face_detection_feature_ptr);
    uint8_t** byte_data = reinterpret_cast<uint8_t**>(output_data);
    results.clear();
    auto ret = face_detection_feature->postprocess(byte_data, results);
    if (ret != ZETIC_MLANGE_FEATURE_SUCCESS)
        return nullptr;
    
    *output_size = results.size();
    return results.data();
}
}
