
cmake_minimum_required(VERSION 3.22.1)

project(whisper_swift_wrapper)

add_library(whisper_swift_wrapper ${LIBRARY_BUILD_TYPE}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/whisper_swift_wrapper.cpp
)

target_include_directories(whisper_swift_wrapper PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(whisper_swift_wrapper
                        whisper_feature)
