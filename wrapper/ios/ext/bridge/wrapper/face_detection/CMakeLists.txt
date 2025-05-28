
cmake_minimum_required(VERSION 3.22.1)

project(face_detection_swift_wrapper)

add_library(face_detection_swift_wrapper ${LIBRARY_BUILD_TYPE}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/face_detection_swift_wrapper.cpp
)

target_include_directories(face_detection_swift_wrapper PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(face_detection_swift_wrapper PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../opencv/src)

target_link_libraries(face_detection_swift_wrapper
                        face_detection_feature
                        entity_feature
                        opencv_swift_wrapper)
