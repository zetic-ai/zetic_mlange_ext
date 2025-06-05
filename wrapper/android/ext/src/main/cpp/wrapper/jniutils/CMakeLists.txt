cmake_minimum_required(VERSION 3.18.1)

project(jniutils)

add_library(jniutils_ext SHARED
        src/memory_manager.cpp
        src/jni_utils.cpp
        src/jni_memory_manager.cpp
)

target_include_directories(jniutils_ext PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/inc)

find_library(log-lib log)

target_link_libraries(jniutils_ext
        android
        log
)
