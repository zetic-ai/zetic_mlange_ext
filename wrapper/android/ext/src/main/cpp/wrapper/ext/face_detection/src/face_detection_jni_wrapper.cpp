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

extern "C"
JNIEXPORT jbyteArray JNICALL
Java_com_zeticai_mlange_feature_facedetection_FaceDetectionWrapper_nativePreprocess(
        JNIEnv *env, jobject thiz, jlong face_detection_feature_ptr,
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

extern "C"
JNIEXPORT jobject JNICALL
Java_com_zeticai_mlange_feature_facedetection_FaceDetectionWrapper_nativePostprocess(
        JNIEnv *env, jobject thiz, jlong face_detection_feature_ptr,
        jobjectArray output_data) {
    FaceDetectionFeature *face_detection_feature = reinterpret_cast<FaceDetectionFeature *>(face_detection_feature_ptr);

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

    std::vector<FaceDetectionResult> face_detection_results;

    Zetic_MLange_Feature_Result_t result = face_detection_feature->postprocess(
            output_raw_data, face_detection_results);

    if (result != ZETIC_MLANGE_FEATURE_SUCCESS)
        return nullptr;

    jclass array_list_class = env->FindClass("java/util/ArrayList");
    jmethodID array_list_constructor = env->GetMethodID(array_list_class, "<init>", "()V");
    jobject array_list_face_detection_result = env->NewObject(array_list_class,
                                                              array_list_constructor);
    jmethodID array_list_add = env->GetMethodID(array_list_class, "add", "(Ljava/lang/Object;)Z");

    jclass face_detection_result_class = env->FindClass(
            "com/zeticai/mlange/feature/facedetection/FaceDetectionResults");
    jmethodID face_detection_result_class_constructor = env->GetMethodID(
            face_detection_result_class,
            "<init>", "(Ljava/util/List;)V");

    for (int i = 0; i < face_detection_results.size(); i++) {
        jobject box = JNIMemoryManager::acquire(env, "com/zeticai/mlange/feature/entity/Box", "(FFFF)V",
                                     face_detection_results[i].bounding_box.x_min,
                                     face_detection_results[i].bounding_box.y_min,
                                     face_detection_results[i].bounding_box.x_max,
                                     face_detection_results[i].bounding_box.y_max);

        jobject face_detection = JNIMemoryManager::acquire(env, "com/zeticai/mlange/feature/facedetection/FaceDetectionResult",
                                                           "(Lcom/zeticai/mlange/feature/entity/Box;F)V",
                                                box, face_detection_results[i].score);

        env->CallBooleanMethod(array_list_face_detection_result, array_list_add, face_detection);
    }

    return env->NewObject(
            face_detection_result_class,
            face_detection_result_class_constructor,
            array_list_face_detection_result);
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_zeticai_mlange_feature_facedetection_FaceDetectionWrapper_nativeInit(JNIEnv *env,
                                                                              jobject thiz) {
    FaceDetectionFeature *face_detection_feature = new FaceDetectionFeature();
    return reinterpret_cast<jlong>(face_detection_feature);
}
extern "C"
JNIEXPORT void JNICALL
Java_com_zeticai_mlange_feature_facedetection_FaceDetectionWrapper_nativeDeinit(JNIEnv *env,
                                                                                jobject thiz,
                                                                                jlong face_detection_feature_ptr) {
    delete reinterpret_cast<FaceDetectionFeature *>(face_detection_feature_ptr);
}
