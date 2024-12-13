cmake_minimum_required(VERSION 3.13)

project(entity_ext)

add_library(entity_ext ${LIBRARY_BUILD_TYPE} inc/box.h inc/landmark.h)

target_include_directories(entity_ext PUBLIC inc)

target_include_directories(entity_ext PUBLIC ${OpenCV_INCLUDE_DIRS})

if(PLATFORM STREQUAL "android")
    target_link_libraries(entity_ext
            ${OpenCV_LIBS}
            opencv_feature
            zetic_feature_common
            m
            android
            log)
else()
    target_link_libraries(entity_ext
            ${OpenCV_LIBS}
            opencv_feature
            zetic_feature_common)
endif()
