#include "face_landmark_swift_wrapper.h"

extern "C" {

static float* blob;
FaceLandmarkResult* face_landmark_result;

long nativeInitFaceLandmark() {
    FaceLandmarkFeature *face_landmark_feature = new FaceLandmarkFeature();
    return reinterpret_cast<long>(face_landmark_feature);
}

void nativeDeinitFaceLandmark(long face_landmark_feature_ptr) {
    delete reinterpret_cast<FaceLandmarkFeature*>(face_landmark_feature_ptr);
}

float* nativePreprocessFaceLandmark(long face_landmark_feature_ptr, CGImageRef input_image, float x_min, float y_min, float x_max, float y_max, int* blob_size) {
    FaceLandmarkFeature* face_landmark_feature = reinterpret_cast<FaceLandmarkFeature*>(face_landmark_feature_ptr);
    cv::Mat img = _getCVMatFromCGImageRef(input_image);
    
    if (img.empty())
        return nullptr;
   
    cv::Mat input_data;
    auto ret = face_landmark_feature->preprocess(img, Box(x_min, y_min, x_max, y_max), input_data);
    if (ret != ZETIC_MLANGE_FEATURE_SUCCESS)
        return nullptr;
    
    size_t len_blob = input_data.total() * 3;
    *blob_size = len_blob;
    if (!blob) {
        blob = new float[len_blob];
    }
    
    MLangeFeatureOpenCV mlange_feature_opencv;
    ret = mlange_feature_opencv.getFlatFloatArrayFromImage(input_data, blob);
    if (ret != ZETIC_MLANGE_FEATURE_SUCCESS)
        return nullptr;
    
    return blob;
}

Landmark* nativePostprocessFaceLandmark(long face_landmark_feature_ptr, float** output_data, float* confidence, int* output_size) {
    FaceLandmarkFeature* face_landmark_feature = reinterpret_cast<FaceLandmarkFeature*>(face_landmark_feature_ptr);
    uint8_t** byte_data = reinterpret_cast<uint8_t**>(output_data);
    
    if (!face_landmark_result)
        face_landmark_result = new FaceLandmarkResult();
    face_landmark_result->landmarks.clear();
    
    auto ret = face_landmark_feature->postprocess(byte_data, *face_landmark_result);
    if (ret != ZETIC_MLANGE_FEATURE_SUCCESS)
        return nullptr;
    
    *confidence = face_landmark_result->confidence;
    *output_size = face_landmark_result->landmarks.size();

    return face_landmark_result->landmarks.data();
}
}
