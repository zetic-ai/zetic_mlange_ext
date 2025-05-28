cmake_minimum_required(VERSION 3.13)

project(opencv_feature)

set(OPENCV_PATH ${THIRD_PARTY_PATH}/opencv)

if(PLATFORM STREQUAL iOS OR PLATFORM STREQUAL iOS-simulator)
    set(OpenCV_DIR "${OPENCV_PATH}/ios_install/lib/cmake/opencv4")

elseif(ANDROID OR PLATFORM STREQUAL android)
    set(OpenCV_DIR "${OPENCV_PATH}/android/sdk/native/jni")

else()
    set(OpenCV_DIR "${OPENCV_PATH}/build/release")
endif()

find_package(OpenCV REQUIRED)

add_library(opencv_feature ${LIBRARY_BUILD_TYPE}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/feature_opencv.cpp
)

target_include_directories(opencv_feature
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/inc
        ${OpenCV_INCLUDE_DIRS}
)

if(PLATFORM STREQUAL iOS OR PLATFORM STREQUAL iOS-simulator)
    target_link_libraries(opencv_feature
            PUBLIC
            common_feature
            ${OpenCV_LIBS}
            ${OpenCV_FRAMEWORKS}
    )

elseif(ANDROID OR PLATFORM STREQUAL android)
    message(OpenCV_LIBS)
    message("${OpenCV_LIBS}")
    target_link_libraries(opencv_feature
            PUBLIC
            common_feature
            ${OpenCV_LIBS}
            m
            android
            log
    )

else()
    target_link_libraries(opencv_feature
            PUBLIC
            common_feature
            ${OpenCV_LIBS}
    )
endif()
