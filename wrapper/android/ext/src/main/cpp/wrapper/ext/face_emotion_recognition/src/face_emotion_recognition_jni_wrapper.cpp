#include <jni.h>
#include <cstdlib>
#include <sstream>
#include <vector>
#include <android/log.h>

#include "dbg_utils.h"

#include <iostream>

#include <android/bitmap.h>

#include "getopt.h"
#include "face_emotion_recognition_feature.h"

using namespace ZeticMLange;

extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_zeticai_mlange_feature_faceemotionrecognition_FaceEmotionRecognitionWrapper_nativePreprocess(
        JNIEnv *env, jobject thiz, jlong face_emotion_recognition_feature_ptr, jlong input_img_ptr,
        jfloat x_min, jfloat y_min, jfloat x_max, jfloat y_max) {
    FaceEmotionRecognitionFeature *face_emotion_recognition_feature = reinterpret_cast<FaceEmotionRecognitionFeature *>(face_emotion_recognition_feature_ptr);
    cv::Mat *img = reinterpret_cast<cv::Mat *> (input_img_ptr);

    if (img->empty())
        return nullptr;

    cv::Mat input_data;
    auto ret = face_emotion_recognition_feature->preprocess(*img, Box(x_min, y_min, x_max, y_max),
                                                            input_data);
    if (ret != ZETIC_MLANGE_FEATURE_SUCCESS)
        return nullptr;

    size_t buffer_size = input_data.total() * input_data.elemSize();
    jbyteArray byte_array = env->NewByteArray(buffer_size);
    if (byte_array != nullptr) {
        env->SetByteArrayRegion(byte_array, 0, buffer_size,
                                reinterpret_cast<jbyte *>(input_data.data));
    }
    return byte_array;
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_zeticai_mlange_feature_faceemotionrecognition_FaceEmotionRecognitionWrapper_nativePostprocess(
        JNIEnv *env, jobject thiz, jlong face_emotion_recognition_feature_ptr,
        jobjectArray output_data) {

    auto *face_emotion_recognition_feature = reinterpret_cast<FaceEmotionRecognitionFeature *>(
            face_emotion_recognition_feature_ptr);

    if (!output_data)
        return nullptr;

    jsize array_length = env->GetArrayLength(output_data);
    if (array_length == 0)
        return nullptr;

    std::vector<uint8_t *> buffer_pointers(array_length);

    for (jsize i = 0; i < array_length; ++i) {
        jobject byte_buffer = env->GetObjectArrayElement(output_data, i);
        if (!byte_buffer)
            return nullptr;

        void *addr = env->GetDirectBufferAddress(byte_buffer);
        jlong capacity = env->GetDirectBufferCapacity(byte_buffer);

        if (!addr || capacity <= 0) {
            env->DeleteLocalRef(byte_buffer);
            return nullptr;
        }

        buffer_pointers[i] = reinterpret_cast<uint8_t *>(addr);
        env->DeleteLocalRef(byte_buffer);
    }

    uint8_t **output_raw_data = buffer_pointers.data();

    std::pair<float, std::string> postprocess_result;

    auto result = face_emotion_recognition_feature->postprocess(output_raw_data,
                                                                postprocess_result);

    if (result != ZETIC_MLANGE_FEATURE_SUCCESS)
        return nullptr;

    jclass result_class = env->FindClass(
            "com/zeticai/mlange/feature/faceemotionrecognition/FaceEmotionRecognitionResult");
    jmethodID result_ctor = env->GetMethodID(result_class, "<init>", "(FLjava/lang/String;)V");

    jstring emotion_str = env->NewStringUTF(postprocess_result.second.c_str());

    return env->NewObject(result_class, result_ctor, postprocess_result.first, emotion_str);
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_zeticai_mlange_feature_faceemotionrecognition_FaceEmotionRecognitionWrapper_nativeInit(
        JNIEnv *env, jobject thiz) {
    FaceEmotionRecognitionFeature *face_emotion_recognition_feature = new FaceEmotionRecognitionFeature();
    return reinterpret_cast<jlong>(face_emotion_recognition_feature);
}
extern "C" JNIEXPORT void JNICALL
Java_com_zeticai_mlange_feature_faceemotionrecognition_FaceEmotionRecognitionWrapper_nativeDeinit(
        JNIEnv *env, jobject thiz, jlong face_emotion_recognition_feature_ptr) {
    delete reinterpret_cast<FaceEmotionRecognitionFeature *>(face_emotion_recognition_feature_ptr);
}