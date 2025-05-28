cmake_minimum_required(VERSION 3.18.1)

project(face_emotion_recognition_jni_wrapper)

add_library(face_emotion_recognition_jni_wrapper SHARED
        src/face_emotion_recognition_jni_wrapper.cpp
)

find_library(log-lib log)
find_library(jnigraphics-lib jnigraphics)

target_link_libraries(face_emotion_recognition_jni_wrapper
        face_emotion_recognition_feature
        android
        log
        ${jnigraphics-lib})
