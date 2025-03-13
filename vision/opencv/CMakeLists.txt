cmake_minimum_required(VERSION 3.13)

project(opencv_feature)

set(ZETIC_UTIL_OPENCV_DIR ${ZETIC_UTIL_PATH}/third-party/opencv-4.10.0)
set(ZETIC_UTIL_OPENCV_SYMLINK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/zetic-opencv-4.10.0)

# Make symlink for zetic_util
file(CREATE_LINK
        ${ZETIC_UTIL_OPENCV_DIR}
        ${ZETIC_UTIL_OPENCV_SYMLINK_DIR}
        SYMBOLIC)

if(PLATFORM STREQUAL iOS OR PLATFORM STREQUAL iOS-simulator)
    # Build for (1) XCode - iOS
    add_library(opencv_feature ${LIBRARY_BUILD_TYPE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/feature_opencv.cpp)
    target_include_directories(opencv_feature PUBLIC
                                ${CMAKE_CURRENT_SOURCE_DIR}/inc)
                                
    set(OPENCV_IOS_INSTALL_PATH "${ZETIC_UTIL_OPENCV_SYMLINK_DIR}/ios_install")
    set(OPENCV_IOS_BUILD_PATH "${ZETIC_UTIL_OPENCV_SYMLINK_DIR}/ios_build")

    include_directories(${OPENCV_IOS_INSTALL_PATH}/include/opencv4)
    target_include_directories(opencv_feature
        PUBLIC
            ${OPENCV_IOS_INSTALL_PATH}/include/opencv4
    )

    target_link_libraries(opencv_feature PUBLIC
        common_feature
    )

elseif(DEFINED ZETIC_MLANGE_ROOT_PATH)
    # Build for (2) Android Studio - Android
    add_library(opencv_feature ${LIBRARY_BUILD_TYPE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/feature_opencv.cpp)

    target_include_directories(opencv_feature PUBLIC
                    ${CMAKE_CURRENT_SOURCE_DIR}/inc)

    set(OPENCV_ANDROID_SDK_PATH "${ZETIC_UTIL_OPENCV_SYMLINK_DIR}/android/sdk")
    set(OpenCV_DIR "${OPENCV_ANDROID_SDK_PATH}/native/jni")
    add_subdirectory(${OPENCV_ANDROID_SDK_PATH}/libcxx_helper)

    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})

    target_include_directories(opencv_feature PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(opencv_feature
                            ${OpenCV_LIBS}
                            common_feature
                            m
                            android
                            log)
else()
    # Build for (3) linux, android - Mentat binaries
    add_library(opencv_feature ${LIBRARY_BUILD_TYPE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/feature_opencv.cpp)

    target_include_directories(opencv_feature PUBLIC
                    ${CMAKE_CURRENT_SOURCE_DIR}/inc)

    if(PLATFORM STREQUAL android)
        set(OPENCV_ANDROID_SDK_PATH "${ZETIC_UTIL_OPENCV_SYMLINK_DIR}/android/sdk")
        set(OpenCV_DIR "${OPENCV_ANDROID_SDK_PATH}/native/jni")
        add_subdirectory(${OPENCV_ANDROID_SDK_PATH}/libcxx_helper)
    else()
        set(OpenCV_DIR "${ZETIC_UTIL_OPENCV_SYMLINK_DIR}/build/release")
    endif()

    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
    target_include_directories(opencv_feature PUBLIC ${OpenCV_INCLUDE_DIRS})

    if(PLATFORM STREQUAL "android")
        target_link_libraries(opencv_feature
                        ${OpenCV_LIBS}
                        common_feature
                        m
                        android
                        log)
    else()
        target_link_libraries(opencv_feature
                        ${OpenCV_LIBS}
                        common_feature)
    endif()
endif()
