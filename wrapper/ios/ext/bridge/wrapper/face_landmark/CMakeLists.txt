cmake_minimum_required(VERSION 3.22.1)

project(face_landmark_swift_wrapper)

add_library(face_landmark_swift_wrapper ${LIBRARY_BUILD_TYPE}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/face_landmark_swift_wrapper.cpp
)

target_include_directories(face_landmark_swift_wrapper PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(face_landmark_swift_wrapper PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../opencv/src)

target_link_libraries(face_landmark_swift_wrapper
                        face_landmark_feature
                        entity_feature
                        ZeticMLangeFeatureOpenCVNative)
