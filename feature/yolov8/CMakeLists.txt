cmake_minimum_required(VERSION 3.13)

project(yolov8_feature)

add_library(yolov8_feature ${LIBRARY_BUILD_TYPE}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/yolov8_feature.cpp)

target_include_directories(yolov8_feature PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/inc)

target_include_directories(yolov8_feature PUBLIC ${OpenCV_INCLUDE_DIRS})

if (PLATFORM STREQUAL "android")
    target_link_libraries(yolov8_feature
            ${OpenCV_LIBS}
            opencv_feature
            zetic_feature_common
            m
            android
            log)

elseif (PLATFORM STREQUAL "ios")
    target_link_libraries(yolov8_feature
            ${OpenCV_LIBS}
            opencv_feature
            zetic_feature_common)
else ()
    target_link_libraries(yolov8_feature
            ${OpenCV_LIBS}
            opencv_feature
            zetic_feature_common)
endif ()

# TODO: Test executables need mlange_ort, mlange_qc. 
# add_executable(run_test_yolov8_ort
#                 ${CMAKE_CURRENT_SOURCE_DIR}/proc/run_test_yolov8_ort.cpp)

# add_executable(run_test_yolov8_qc
#             ${CMAKE_CURRENT_SOURCE_DIR}/proc/run_test_yolov8_qc.cpp)

# target_link_libraries(run_test_yolov8_ort zetic_mlange_feature_yolov8)
# target_link_libraries(run_test_yolov8_qc zetic_mlange_feature_yolov8)
