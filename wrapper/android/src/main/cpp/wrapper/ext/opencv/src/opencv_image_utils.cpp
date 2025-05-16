#include <jni.h>

#include <android/native_window_jni.h>
#include "feature_opencv.h"

cv::Mat current_image;

extern "C"
JNIEXPORT void JNICALL
Java_com_zeticai_mlange_feature_vision_OpenCVImageUtilsWrapper_nativeSetSurface(JNIEnv *env,
                                                                                jobject thiz,
                                                                                jobject surface) {
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_zeticai_mlange_feature_vision_OpenCVImageUtilsWrapper_nativeFrame(JNIEnv *env,
                                                                           jobject thiz,
                                                                           jbyteArray image,
                                                                           jint rotate) {
    jbyte *jpeg_data = env->GetByteArrayElements(image, nullptr);
    jsize length = env->GetArrayLength(image);

    std::vector<uchar> imgBuffer(jpeg_data, jpeg_data + length);

    current_image = cv::imdecode(imgBuffer, cv::IMREAD_COLOR);

    if (rotate == 90) {
        cv::rotate(current_image, current_image, cv::RotateFlags::ROTATE_90_CLOCKWISE);
    } else if (rotate == -90) {
        cv::rotate(current_image, current_image, cv::RotateFlags::ROTATE_90_COUNTERCLOCKWISE);
    }

    env->ReleaseByteArrayElements(image, jpeg_data, 0);

    return reinterpret_cast<jlong>(&current_image);
}
