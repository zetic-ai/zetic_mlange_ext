#include <jni.h>
#include <cstdlib>
#include <sstream>
#include <vector>
#include <android/log.h>

#include "dbg_utils.h"
#include "jni_memory_manager.h"

#include <iostream>

#include "getopt.h"
#include "face_detection_feature.h"

using namespace ZeticMLange;

extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_zeticai_mlange_feature_facedetection_FaceDetectionWrapper_nativePreprocess(JNIEnv *env,
                                                                                    jobject thiz,
                                                                                    jlong face_detection_feature_ptr,
                                                                                    jlong input_img_ptr) {
    FaceDetectionFeature *face_detection_feature = reinterpret_cast<FaceDetectionFeature *>(face_detection_feature_ptr);
    cv::Mat *img = reinterpret_cast<cv::Mat *> (input_img_ptr);

    if (img->empty())
        return nullptr;

    cv::Mat input_data;
    auto ret = face_detection_feature->preprocess(*img, input_data);
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
Java_com_zeticai_mlange_feature_facedetection_FaceDetectionWrapper_nativePostprocess(JNIEnv *env,
                                                                                     jobject thiz,
                                                                                     jlong face_detection_feature_ptr,
                                                                                     jobjectArray output_data) {

    auto *face_detection_feature = reinterpret_cast<FaceDetectionFeature *>(face_detection_feature_ptr);

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

    std::vector<FaceDetectionResult> face_detection_results;
    auto result = face_detection_feature->postprocess(output_raw_data, face_detection_results);

    if (result != ZETIC_MLANGE_FEATURE_SUCCESS)
        return nullptr;

    jclass array_list_class = env->FindClass("java/util/ArrayList");
    jmethodID array_list_ctor = env->GetMethodID(array_list_class, "<init>", "()V");
    jobject result_list = env->NewObject(array_list_class, array_list_ctor);

    jmethodID array_list_add = env->GetMethodID(array_list_class, "add", "(Ljava/lang/Object;)Z");

    jclass face_detection_results_class = env->FindClass(
            "com/zeticai/mlange/feature/facedetection/FaceDetectionResults");
    jmethodID results_ctor = env->GetMethodID(face_detection_results_class, "<init>",
                                              "(Ljava/util/List;)V");

    for (const auto &r: face_detection_results) {

        jobject box = JNIMemoryManager::acquire(env, "com/zeticai/mlange/feature/entity/Box",
                                                "(FFFF)V", r.bounding_box.x_min,
                                                r.bounding_box.y_min, r.bounding_box.x_max,
                                                r.bounding_box.y_max);

        jobject face_detection = JNIMemoryManager::acquire(env,
                                                           "com/zeticai/mlange/feature/facedetection/FaceDetectionResult",
                                                           "(Lcom/zeticai/mlange/feature/entity/Box;F)V",
                                                           box, r.score);

        env->CallBooleanMethod(result_list, array_list_add, face_detection);
    }

    return env->NewObject(face_detection_results_class, results_ctor, result_list);
}


extern "C" JNIEXPORT jlong JNICALL
Java_com_zeticai_mlange_feature_facedetection_FaceDetectionWrapper_nativeInit(JNIEnv *env,
                                                                              jobject thiz) {
    FaceDetectionFeature *face_detection_feature = new FaceDetectionFeature();
    return reinterpret_cast<jlong>(face_detection_feature);
}
extern "C" JNIEXPORT void JNICALL
Java_com_zeticai_mlange_feature_facedetection_FaceDetectionWrapper_nativeDeinit(JNIEnv *env,
                                                                                jobject thiz,
                                                                                jlong face_detection_feature_ptr) {
    delete reinterpret_cast<FaceDetectionFeature *>(face_detection_feature_ptr);
}
