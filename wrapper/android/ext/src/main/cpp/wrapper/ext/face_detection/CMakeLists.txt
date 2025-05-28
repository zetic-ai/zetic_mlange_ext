cmake_minimum_required(VERSION 3.18.1)

project(face_detection_jni_wrapper)

add_library(face_detection_jni_wrapper SHARED
        src/face_detection_jni_wrapper.cpp
)

find_library(log-lib log)
find_library(jnigraphics-lib jnigraphics)

target_link_libraries(face_detection_jni_wrapper
        face_detection_feature
        android
        jniutils_ext
        log
        ${jnigraphics-lib})
