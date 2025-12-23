cmake_minimum_required(VERSION 3.5)

project(common_feature C)

string(TOLOWER ${CMAKE_BUILD_TYPE} BUILD_TYPE)
if (${BUILD_TYPE} STREQUAL debug)
    add_compile_definitions(ENABLE_ASSERT=1)
endif()

set(ZETIC_FEATURE_COMMON_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/inc)

set(ZETIC_FEATURE_COMMON_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/zetic_feature_types.c)

add_library(common_feature ${LIBRARY_BUILD_TYPE} ${ZETIC_FEATURE_COMMON_SRCS})
target_include_directories(common_feature PUBLIC ${ZETIC_FEATURE_COMMON_INCLUDE_DIRS})

if(PLATFORM STREQUAL "android")
    target_link_libraries(common_feature PUBLIC dl android log)
else()
    target_link_libraries(common_feature PUBLIC dl)
endif()
