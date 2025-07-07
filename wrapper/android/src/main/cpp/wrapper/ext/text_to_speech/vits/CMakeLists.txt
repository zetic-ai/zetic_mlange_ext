cmake_minimum_required(VERSION 3.18.1)

project(vits_jni_wrapper)

add_library(vits_jni_wrapper SHARED src/vits_jni_wrapper.cpp)

find_library(log-lib log)

target_link_libraries(vits_jni_wrapper PRIVATE
        vits_feature
        jniutils::jniutils
        android
        log)
