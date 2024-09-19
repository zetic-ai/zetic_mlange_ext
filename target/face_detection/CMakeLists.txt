cmake_minimum_required(VERSION 3.13)

project(zetic_mlange_feature_face_detection)

add_library(zetic_mlange_feature_face_detection SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/src/face_detection_feature_opencv.cpp
)

target_include_directories(zetic_mlange_feature_face_detection PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/inc)

target_include_directories(zetic_mlange_feature_face_detection PUBLIC ${OpenCV_INCLUDE_DIRS})

if(PLATFORM STREQUAL "android")
    target_link_libraries(zetic_mlange_feature_face_detection
            ${OpenCV_LIBS}
            zetic_mlange_feature_opencv
            zetic_feature_common
            m
            android
            log)
elseif(PLATFORM STREQUAL "ios")
    target_link_libraries(zetic_mlange_feature_face_detection
            ${OpenCV_LIBS}
            zetic_mlange_feature_opencv
            zetic_feature_common)
endif()
