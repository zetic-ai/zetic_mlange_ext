
cmake_minimum_required(VERSION 3.22.1)

project(yolov8_swift_wrapper)

add_library(yolov8_swift_wrapper ${LIBRARY_BUILD_TYPE}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/yolov8_swift_wrapper.cpp)

target_include_directories(yolov8_swift_wrapper PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(yolov8_swift_wrapper PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../opencv/src)

target_link_libraries(yolov8_swift_wrapper
                        yolov8_feature
                        opencv_swift_wrapper)
