cmake_minimum_required(VERSION 3.01)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "-g")
    set(CMAKE_C_FLAGS_DEBUG "-g")
endif()

set(ZETIC_UTIL_PATH "$ENV{HOME}/zetic_utils")

if(NOT ZETIC_MLANGE_TARGET)
    set(ZETIC_MLANGE_TARGET x86_64-linux-clang)
endif()

if(ZETIC_MLANGE_TARGET STREQUAL x86_64-linux-clang)
    set(PLATFORM "linux-x86-64")
    add_compile_options(-march=x86-64)
elseif(ZETIC_MLANGE_TARGET STREQUAL aarch64-android)
    set(PLATFORM "android")
    set(CMAKE_ANDROID_ARCH_ABI arm64-v8a)
    set(ANDROID_ABI arm64-v8a)
elseif(ZETIC_MLANGE_TARGET STREQUAL arm-android)
    set(PLATFORM "android")
    set(CMAKE_ANDROID_ARCH_ABI armeabi-v7a)
    set(ANDROID_ABI armeabi-v7a)
elseif(DEFINED ANDROID_ABI)
    set(PLATFORM "android")
    set(CMAKE_ANDROID_ARCH_ABI ${ANDROID_ABI})
elseif(ZETIC_MLANGE_TARGET STREQUAL ios)
    set(PLATFORM "ios")
else()
    message(FATAL_ERROR "[ERROR] INVALID ZETIC_MLANGE_TARGET")
endif()

if(PLATFORM STREQUAL android)
    set(ANDROID_NDK $ENV{ANDROID_NDK})
    if(NOT ANDROID_NDK)
        message(FATAL_ERROR "[ERROR] ANDROID_NDK Not set!")
    else()
        set(CMAKE_ANDROID_NDK ${ANDROID_NDK})
        message("ANDROID_NDK: ${ANDROID_NDK}")
    endif()
endif()

if(PLATFORM STREQUAL android)
    set(CMAKE_TOOLCHAIN_FILE ${ANDROID_NDK}/build/cmake/android.toolchain.cmake)
    set(ANDROID_PLATFORM android-28)
    message("ANDROID_TOOL_CHAIN: ${CMAKE_TOOLCHAIN_FILE}")
endif()

project(zetic_mlange_ext CXX)

add_compile_options(-Werror)

add_subdirectory(common)

add_subdirectory(vision)

add_subdirectory(tokenizer)

# Target should be included at last
add_subdirectory(target)
