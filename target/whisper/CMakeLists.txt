cmake_minimum_required(VERSION 3.13)

project(zetic_mlange_feature_whisper)

add_library(zetic_mlange_feature_whisper ${LIBRARY_BUILD_TYPE}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/whisper_processor.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/whisper_tokenizer.cpp
)

target_include_directories(zetic_mlange_feature_whisper PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/inc)

if(PLATFORM STREQUAL "android")
    target_link_libraries(zetic_mlange_feature_whisper
            zetic_feature_common
            m
            android
            log)
else()
    target_link_libraries(zetic_mlange_feature_whisper
            zetic_feature_common)
endif()
