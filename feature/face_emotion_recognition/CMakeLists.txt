cmake_minimum_required(VERSION 3.13)

project(face_emotion_recognition_feature)

add_library(face_emotion_recognition_feature ${LIBRARY_BUILD_TYPE}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/face_emotion_recognition_feature.cpp
)

target_include_directories(face_emotion_recognition_feature PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/inc)

target_include_directories(face_emotion_recognition_feature PUBLIC ${OpenCV_INCLUDE_DIRS})

if (PLATFORM STREQUAL "android")
    target_link_libraries(face_emotion_recognition_feature
            ${OpenCV_LIBS}
            opencv_feature
            zetic_feature_common
            m
            android
            log)
else ()
    target_link_libraries(face_emotion_recognition_feature
            ${OpenCV_LIBS}
            opencv_feature
            zetic_feature_common)
endif ()
