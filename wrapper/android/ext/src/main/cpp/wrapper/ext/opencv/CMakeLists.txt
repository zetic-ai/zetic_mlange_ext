cmake_minimum_required(VERSION 3.18.1)

project(opencv_jni_wrapper)

add_library(opencv_jni_wrapper SHARED
        src/opencv_image_utils.cpp
)

find_library(log-lib log)

target_link_libraries(opencv_jni_wrapper
        opencv_feature
        android
        log)
