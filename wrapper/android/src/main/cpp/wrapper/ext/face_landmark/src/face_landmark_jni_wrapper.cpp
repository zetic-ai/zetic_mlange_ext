#include <jni.h>
#include <cstdlib>
#include <sstream>
#include <vector>
#include <android/log.h>

#include "dbg_utils.h"

#include <iostream>

#include <android/bitmap.h>

#include "getopt.h"
#include "face_landmark_feature.h"

using namespace ZeticMLange;

extern "C"
JNIEXPORT jbyteArray JNICALL
Java_com_zeticai_mlange_feature_facelandmark_FaceLandmarkWrapper_nativePreprocess(
        JNIEnv *env, jobject thiz, jlong face_landmark_feature_ptr,
        jlong input_img_ptr, jfloat x_min, jfloat y_min, jfloat x_max, jfloat y_max) {
    FaceLandmarkFeature *face_landmark_feature = reinterpret_cast<FaceLandmarkFeature *>(face_landmark_feature_ptr);
    cv::Mat *img = reinterpret_cast<cv::Mat *> (input_img_ptr);

    if (img->empty())
        return nullptr;

    cv::Mat input_data;
    auto ret = face_landmark_feature->preprocess(*img, Box(x_min, y_min, x_max, y_max), input_data);
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

extern "C"
JNIEXPORT jobject JNICALL
Java_com_zeticai_mlange_feature_facelandmark_FaceLandmarkWrapper_nativePostprocess(
        JNIEnv *env, jobject thiz, jlong face_emotion_recognition_feature_ptr,
        jobjectArray output_data) {
    FaceLandmarkFeature *face_landmark_feature = reinterpret_cast<FaceLandmarkFeature *>(face_emotion_recognition_feature_ptr);

    jsize array_length = env->GetArrayLength(output_data);

    std::vector<uint8_t *> buffer_pointers(array_length);

    for (jsize i = 0; i < array_length; ++i) {
        jbyteArray byte_array = (jbyteArray) env->GetObjectArrayElement(output_data, i);

        jsize length = env->GetArrayLength(byte_array);
        jbyte *byte_ptr = env->GetByteArrayElements(byte_array, nullptr);

        buffer_pointers[i] = reinterpret_cast<uint8_t *>(byte_ptr);

        env->ReleaseByteArrayElements(byte_array, byte_ptr, JNI_ABORT);
        env->DeleteLocalRef(byte_array);
    }

    uint8_t **output_raw_data = buffer_pointers.data();

    FaceLandmarkResult face_landmark_result;

    Zetic_MLange_Feature_Result_t result = face_landmark_feature->postprocess(
            output_raw_data, face_landmark_result);

    if (result != ZETIC_MLANGE_FEATURE_SUCCESS)
        return nullptr;

    jclass face_landmark_result_class = env->FindClass(
            "com/zeticai/mlange/feature/facelandmark/FaceLandmarkResult");
    jmethodID face_landmark_result_class_constructor = env->GetMethodID(face_landmark_result_class,
                                                                        "<init>",
                                                                        "(Ljava/util/List;F)V");

    jclass landmark_class = env->FindClass("com/zeticai/mlange/feature/entity/Landmark");
    jmethodID landmark_class_constructor = env->GetMethodID(landmark_class, "<init>", "(FFF)V");

    jclass array_list_class = env->FindClass("java/util/ArrayList");
    jmethodID array_list_constructor = env->GetMethodID(array_list_class, "<init>", "()V");
    jobject array_list_face_landmarks = env->NewObject(array_list_class, array_list_constructor);
    jmethodID array_list_add = env->GetMethodID(array_list_class, "add", "(Ljava/lang/Object;)Z");

    for (size_t i = 0; i < face_landmark_result.landmarks.size(); i++) {
        jobject landmark = env->NewObject(landmark_class, landmark_class_constructor,
                                          face_landmark_result.landmarks[i].x,
                                          face_landmark_result.landmarks[i].y,
                                          face_landmark_result.landmarks[i].z);
        env->CallBooleanMethod(array_list_face_landmarks, array_list_add, landmark);
        env->DeleteLocalRef(landmark);
    }

    return env->NewObject(face_landmark_result_class,
                          face_landmark_result_class_constructor,
                          array_list_face_landmarks,
                          face_landmark_result.confidence);
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_zeticai_mlange_feature_facelandmark_FaceLandmarkWrapper_nativeInit(JNIEnv *env,
                                                                            jobject thiz) {
    FaceLandmarkFeature *face_landmark_feature = new FaceLandmarkFeature();
    return reinterpret_cast<jlong>(face_landmark_feature);
}
extern "C"
JNIEXPORT void JNICALL
Java_com_zeticai_mlange_feature_facelandmark_FaceLandmarkWrapper_nativeDeinit(JNIEnv *env,
                                                                              jobject thiz,
                                                                              jlong face_landmark_feature_ptr) {
    delete reinterpret_cast<FaceLandmarkFeature *>(face_landmark_feature_ptr);
}
