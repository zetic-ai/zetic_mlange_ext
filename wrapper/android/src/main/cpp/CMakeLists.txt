cmake_minimum_required(VERSION 3.18.1)

project(zetic_mlange_ext_android)

set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

set(ZETIC_MLANGE_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

set(ZETIC_MLANGE_TARGET ${ANDROID_ABI})

set(LIBRARY_BUILD_TYPE SHARED)

find_package(jniutils REQUIRED CONFIG)

add_subdirectory(zetic_mlange_ext)
add_subdirectory(wrapper/ext)
