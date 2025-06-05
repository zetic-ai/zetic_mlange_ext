
cmake_minimum_required(VERSION 3.22.1)

project(opencv_swift_wrapper)

add_library(opencv_swift_wrapper ${LIBRARY_BUILD_TYPE}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/opencv_swift_wrapper.cpp)

target_include_directories(opencv_swift_wrapper PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(opencv_swift_wrapper
            opencv_feature)
