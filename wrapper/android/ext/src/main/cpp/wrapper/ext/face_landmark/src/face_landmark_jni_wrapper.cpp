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

extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_zeticai_mlange_feature_facelandmark_FaceLandmarkWrapper_nativePreprocess(JNIEnv *env,
                                                                                  jobject thiz,
                                                                                  jlong face_landmark_feature_ptr,
                                                                                  jlong input_img_ptr,
                                                                                  jfloat x_min,
                                                                                  jfloat y_min,
                                                                                  jfloat x_max,
                                                                                  jfloat y_max) {
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

extern "C" JNIEXPORT jobject JNICALL
Java_com_zeticai_mlange_feature_facelandmark_FaceLandmarkWrapper_nativePostprocess(JNIEnv *env,
                                                                                   jobject thiz,
                                                                                   jlong face_landmark_feature_ptr,
                                                                                   jobjectArray output_data) {

    auto *face_landmark_feature = reinterpret_cast<FaceLandmarkFeature *>(face_landmark_feature_ptr);

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

    FaceLandmarkResult face_landmark_result;
    auto result = face_landmark_feature->postprocess(output_raw_data, face_landmark_result);

    if (result != ZETIC_MLANGE_FEATURE_SUCCESS)
        return nullptr;

    jclass result_class = env->FindClass(
            "com/zeticai/mlange/feature/facelandmark/FaceLandmarkResult");
    jmethodID result_ctor = env->GetMethodID(result_class, "<init>", "(Ljava/util/List;F)V");

    jclass landmark_class = env->FindClass("com/zeticai/mlange/feature/entity/Landmark");
    jmethodID landmark_ctor = env->GetMethodID(landmark_class, "<init>", "(FFF)V");

    jclass array_list_class = env->FindClass("java/util/ArrayList");
    jmethodID array_list_ctor = env->GetMethodID(array_list_class, "<init>", "()V");
    jobject landmark_list = env->NewObject(array_list_class, array_list_ctor);

    jmethodID array_list_add = env->GetMethodID(array_list_class, "add", "(Ljava/lang/Object;)Z");

    for (const auto &lm: face_landmark_result.landmarks) {
        jobject landmark = env->NewObject(landmark_class, landmark_ctor, lm.x, lm.y, lm.z);

        env->CallBooleanMethod(landmark_list, array_list_add, landmark);
        env->DeleteLocalRef(landmark);
    }

    return env->NewObject(result_class, result_ctor, landmark_list,
                          face_landmark_result.confidence);
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_zeticai_mlange_feature_facelandmark_FaceLandmarkWrapper_nativeInit(JNIEnv *env,
                                                                            jobject thiz) {
    FaceLandmarkFeature *face_landmark_feature = new FaceLandmarkFeature();
    return reinterpret_cast<jlong>(face_landmark_feature);
}
extern "C" JNIEXPORT void JNICALL
Java_com_zeticai_mlange_feature_facelandmark_FaceLandmarkWrapper_nativeDeinit(JNIEnv *env,
                                                                              jobject thiz,
                                                                              jlong face_landmark_feature_ptr) {
    delete reinterpret_cast<FaceLandmarkFeature *>(face_landmark_feature_ptr);
}
