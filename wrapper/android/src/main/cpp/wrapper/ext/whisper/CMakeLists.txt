cmake_minimum_required(VERSION 3.18.1)

project(whisper_jni_wrapper)

add_library(whisper_jni_wrapper SHARED src/whisper_jni_wrapper.cpp)

find_library(log-lib log)

target_link_libraries(whisper_jni_wrapper PRIVATE
        whisper_feature
        jniutils::jniutils
        android
        log)
