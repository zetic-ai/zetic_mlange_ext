#include "face_emotion_recognition_swift_wrapper.h"

extern "C" {

static float* blob;

long nativeInitFaceEmotionRecognition() {
    FaceEmotionRecognitionFeature* face_emotion_recognition_feature = new FaceEmotionRecognitionFeature();
    return reinterpret_cast<long>(face_emotion_recognition_feature);
}

void nativeDeinitFaceEmotionRecognition(long face_emotion_recognition_feature_ptr) {
    delete reinterpret_cast<FaceEmotionRecognitionFeature*>(face_emotion_recognition_feature_ptr);
}

float* nativePreprocessFaceEmotionRecognition(long face_emotion_recognition_feature_ptr, CGImageRef input_image, float x_min, float y_min, float x_max, float y_max, int* blob_size) {
    FaceEmotionRecognitionFeature* faceEmotionRecognitionFeature = reinterpret_cast<FaceEmotionRecognitionFeature*>(face_emotion_recognition_feature_ptr);
    cv::Mat img = _getCVMatFromCGImageRef(input_image);
    
    if (img.empty())
        return nullptr;
    
    cv::Mat input_data;
    auto ret = faceEmotionRecognitionFeature->preprocess(img, Box(x_min, y_min, x_max, y_max), input_data);
    if (ret != ZETIC_MLANGE_FEATURE_SUCCESS)
        return nullptr;
    
    int len_blob = input_data.total();
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

void nativePostprocessFaceEmotionRecognition(long face_emotion_recognition_feature_ptr, float** output_data, char* emotion, float* confidence) {
    FaceEmotionRecognitionFeature* face_emotion_recognition_feature = reinterpret_cast<FaceEmotionRecognitionFeature*>(face_emotion_recognition_feature_ptr);
    uint8_t** byte_data = reinterpret_cast<uint8_t**>(output_data);
    
    std::pair<float, std::string> postprocess_result;
    auto ret = face_emotion_recognition_feature->postprocess(byte_data, postprocess_result);
    if (ret != ZETIC_MLANGE_FEATURE_SUCCESS)
        return nullptr;
    
    *confidence = postprocess_result.first;
    strncpy(emotion, postprocess_result.second.c_str(), postprocess_result.second.size());
}
}
