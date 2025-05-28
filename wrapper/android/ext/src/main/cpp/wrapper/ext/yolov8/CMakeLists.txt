cmake_minimum_required(VERSION 3.18.1)

project(yolov8_jni_wrapper)

add_library(yolov8_jni_wrapper SHARED src/yolov8_jni_wrapper.cpp)

find_library(log-lib log)
# Link the jnigraphics library (needed for AndroidBitmap_* functions)
find_library(jnigraphics-lib jnigraphics)

target_link_libraries(yolov8_jni_wrapper
        yolov8_feature
        jniutils_ext
        android
        log
        ${jnigraphics-lib})
