#include <jni.h>
#include <cstdlib>
#include <sstream>
#include <vector>
#include <android/log.h>

#include "dbg_utils.h"
#include "jni_utils.h"

#include <iostream>

#include <android/bitmap.h>
#include <jni_memory_manager.h>

#include "getopt.h"
#include "yolov8_feature.h"

#define YOLO_NUM_MODEL_INPUT 1
#define YOLO_NUM_MODEL_OUTPUT 1

// TODO: Currently use as static to reduce conversion time from bitmap to cv::mat
static cv::Mat *img;
static int8_t *blob;
static int8_t *byte_array;

extern "C" jbyteArray convertMatToJByteArray(JNIEnv *env, ZeticMLangeYoloV8Feature* yolo_v8_feature, cv::Mat& mat) {
    int len_blob = mat.total() * mat.channels() * sizeof(float);
    if (!blob) {
        blob = new int8_t[len_blob];
    }

    auto ret = yolo_v8_feature->getByteArrayFromImage(mat, blob);
    if (ret != ZETIC_MLANGE_FEATURE_SUCCESS) {
        printf("Failed to get blob data from pre-processed image for ZeticMLangeYolov8");
        return nullptr;
    }

    // Create a new jfloatArray
    jbyteArray result = env->NewByteArray(len_blob);

    // Check if the array was created successfully
    if (result == nullptr) {
        return nullptr; // Out of memory error thrown
    }

    // Set the array elements
    env->SetByteArrayRegion(result, 0, len_blob, reinterpret_cast<const jbyte *>(blob));
    return result;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_zeticai_mlange_feature_yolov8_YOLOv8Wrapper_nativeInitDetect(JNIEnv *env,
                                                                      jobject obj,
                                                                      jstring j_coco_file_path) {

    std::string coco_file_path = convertJStringToCString(env, j_coco_file_path);
    ZeticMLangeYoloV8Feature *yolo_v8_feature = new ZeticMLangeYoloV8Feature(YOLO_DETECT_V8,
                                                                             coco_file_path.c_str());
    return reinterpret_cast<jlong>(yolo_v8_feature);
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_zeticai_mlange_feature_yolov8_YOLOv8Wrapper_nativeInitClassifier(JNIEnv *env,
                                                                          jobject obj,
                                                                          jstring j_coco_file_path) {

    std::string coco_file_path = convertJStringToCString(env, j_coco_file_path);
    ZeticMLangeYoloV8Feature *yolo_v8_feature = new ZeticMLangeYoloV8Feature(YOLO_CLS,
                                                                             coco_file_path.c_str());
    return reinterpret_cast<jlong>(yolo_v8_feature);
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_zeticai_mlange_feature_yolov8_YOLOv8Wrapper_nativePreprocess(JNIEnv *env,
                                                                      jobject obj,
                                                                      jlong yolo_v8_feature_ptr,
                                                                      jlong input_img_ptr) {

    ZeticMLangeYoloV8Feature *yolo_v8_feature = reinterpret_cast<ZeticMLangeYoloV8Feature *>(yolo_v8_feature_ptr);

    if (yolo_v8_feature == nullptr)
        return nullptr;

    Zetic_MLange_Feature_Result_t ret = ZETIC_MLANGE_FEATURE_FAIL;
    cv::Mat *img = reinterpret_cast<cv::Mat *> (input_img_ptr);

    if (img == nullptr)
        return nullptr;

    if (img->empty())
        return nullptr;

    cv::Mat processed_img;
    ret = yolo_v8_feature->preprocess(*img, processed_img);
    if (ret != ZETIC_MLANGE_FEATURE_SUCCESS) {
        printf("Failed to preprocess input image for ZeticMLangeYolov8");
        return nullptr;
    }

    return convertMatToJByteArray(env, yolo_v8_feature, processed_img);
}

extern "C"
JNIEXPORT jobject JNICALL
Java_com_zeticai_mlange_feature_yolov8_YOLOv8Wrapper_nativePostProcess(JNIEnv *env,
                                                                       jobject /* this */,
                                                                       jlong yolo_v8_feature_ptr,
                                                                       jbyteArray output) {

    ZeticMLangeYoloV8Feature *yolo_v8_feature = reinterpret_cast<ZeticMLangeYoloV8Feature *>(yolo_v8_feature_ptr);

    if (yolo_v8_feature == nullptr)
        return nullptr;

    Zetic_MLange_Feature_Result_t ret = ZETIC_MLANGE_FEATURE_FAIL;

    std::vector<DL_RESULT> res;

    // Get the length of the jfloatArray
    jsize length = env->GetArrayLength(output);

    // Allocate a native float array
    if (!byte_array) {
        byte_array = new int8_t[length];
    }

    // Get the elements from the jfloatArray
    env->GetByteArrayRegion(output, 0, length, byte_array);


    ret = yolo_v8_feature->postprocess(res, (void *) byte_array);
    if (ret != ZETIC_MLANGE_FEATURE_SUCCESS) {
        printf("Failed to postprocess!");
        return nullptr;
    }

    jclass array_list_class = env->FindClass("java/util/ArrayList");
    jmethodID array_list_constructor = env->GetMethodID(array_list_class, "<init>", "()V");
    jobject array_list_face_detection_result = env->NewObject(array_list_class,
                                                              array_list_constructor);
    jmethodID array_list_add = env->GetMethodID(array_list_class, "add", "(Ljava/lang/Object;)Z");

    jclass yolo_result_class = env->FindClass(
            "com/zeticai/mlange/feature/yolov8/YOLOResult");
    jmethodID yolo_result_class_constructor = env->GetMethodID(yolo_result_class,
                                                               "<init>", "(Ljava/util/List;)V");

    JNIMemoryManager::clear("com/zeticai/mlange/feature/entity/Box");
    JNIMemoryManager::clear("com/zeticai/mlange/feature/yolov8/YOLOObject");

    for (int i = 0; i < res.size(); i++) {
        jobject box = JNIMemoryManager::acquire(env, "com/zeticai/mlange/feature/entity/Box", "(FFFF)V",
                                     (float) res[i].box.x,
                                     (float) res[i].box.y,
                                     (float) res[i].box.x + (float) res[i].box.width,
                                     (float) res[i].box.y + (float) res[i].box.height);

        jobject yolo_object = JNIMemoryManager::acquire(env,   "com/zeticai/mlange/feature/yolov8/YOLOObject", "(IFLcom/zeticai/mlange/feature/entity/Box;)V",
                                             res[i].class_id, res[i].confidence, box);

        env->CallBooleanMethod(array_list_face_detection_result, array_list_add, yolo_object);
    }

    return env->NewObject(
            yolo_result_class,
            yolo_result_class_constructor,
            array_list_face_detection_result);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_zeticai_mlange_feature_yolov8_YOLOv8Wrapper_nativeFreePreprocessedBuffer(JNIEnv *env,
                                                                                  jobject /* this */,
                                                                                  jobject byte_buffer) {
    // Get the pointer to the native memory from the ByteBuffer
    void *buffer = env->GetDirectBufferAddress(byte_buffer);
    if (buffer != nullptr) {
        free(buffer);  // Free the native memory
    }
}

extern "C"
JNIEXPORT void JNICALL
Java_com_zeticai_mlange_feature_yolov8_YOLOv8Wrapper_nativeDeinit(JNIEnv *env,
                                                                  jobject /* this */,
                                                                  jlong yolov8_feature_ptr) {

    ZeticMLangeYoloV8Feature *yolo_v8_feature = reinterpret_cast<ZeticMLangeYoloV8Feature *>(yolov8_feature_ptr);
    delete (yolo_v8_feature);
    delete (blob);
    delete (byte_array);
}

extern "C"
JNIEXPORT jbyteArray JNICALL
Java_com_zeticai_mlange_feature_yolov8_YOLOv8Wrapper_nativePreprocessWithFrame(JNIEnv *env, jobject thiz,
                                                                               jlong yolov8_feature_ptr,
                                                                               jbyteArray frame,
                                                                               jint width, jint height,
                                                                               jint format_code) {

    ZeticMLangeYoloV8Feature *yolo_v8_feature = reinterpret_cast<ZeticMLangeYoloV8Feature *>(yolov8_feature_ptr);
    jbyte* nvPtr = env->GetByteArrayElements(frame, nullptr);
    if (!nvPtr) {
        return JNI_FALSE;
    }

    cv::Mat bgrMat = MLangeFeatureOpenCV::convertToBGR(
            reinterpret_cast<const uint8_t*>(nvPtr),
            width,
            height,
            format_code
    );

    cv::Mat output;

    yolo_v8_feature->preprocess(bgrMat, output);

    env->ReleaseByteArrayElements(frame, nvPtr, 0);

    return convertMatToJByteArray(env, yolo_v8_feature, output);
}