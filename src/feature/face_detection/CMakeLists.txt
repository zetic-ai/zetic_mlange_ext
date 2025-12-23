cmake_minimum_required(VERSION 3.5)

project(face_detection_feature)

add_library(face_detection_feature ${LIBRARY_BUILD_TYPE}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/face_detection_feature.cpp
)

target_include_directories(face_detection_feature PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/inc)

target_include_directories(face_detection_feature PUBLIC ${OpenCV_INCLUDE_DIRS})

if (PLATFORM STREQUAL "android")
    target_link_libraries(face_detection_feature
            ${OpenCV_LIBS}
            opencv_feature
            common_feature
            entity_feature
            m
            android
            log)
else ()
    target_link_libraries(face_detection_feature
            ${OpenCV_LIBS}
            opencv_feature
            common_feature
            entity_feature)
endif ()
