cmake_minimum_required(VERSION 3.13)

project(whisper_feature)

add_library(whisper_feature ${LIBRARY_BUILD_TYPE}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/whisper_processor.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/whisper_tokenizer.cpp
)

target_include_directories(whisper_feature PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/inc)

if (PLATFORM STREQUAL "android")
    target_link_libraries(whisper_feature
            zetic_feature_common
            m
            android
            log)
else ()
    target_link_libraries(whisper_feature
            zetic_feature_common)
endif ()
