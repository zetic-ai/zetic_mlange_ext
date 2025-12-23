cmake_minimum_required(VERSION 3.13)

project(vits_feature)

add_library(vits_feature ${LIBRARY_BUILD_TYPE}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/vits_tokenizer.cpp
)

target_include_directories(vits_feature PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/inc)

if (PLATFORM STREQUAL "android")
    target_link_libraries(vits_feature
            common_feature
            m
            android
            log)
else ()
    target_link_libraries(vits_feature
            common_feature)
endif ()