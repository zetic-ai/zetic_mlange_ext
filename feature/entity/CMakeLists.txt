cmake_minimum_required(VERSION 3.13)

project(entity_feature)

add_library(entity_feature ${LIBRARY_BUILD_TYPE} src/box.cpp src/landmark.cpp)

target_include_directories(entity_feature PUBLIC inc)

target_include_directories(entity_feature PUBLIC ${OpenCV_INCLUDE_DIRS})

if(PLATFORM STREQUAL "android")
    target_link_libraries(entity_feature
            ${OpenCV_LIBS}
            opencv_feature
            common_feature
            m
            android
            log)
else()
    target_link_libraries(entity_feature
            ${OpenCV_LIBS}
            opencv_feature
            common_feature)
endif()
