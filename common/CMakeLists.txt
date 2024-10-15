cmake_minimum_required(VERSION 3.01)

project(zetic_feature_common C)

string(TOLOWER ${CMAKE_BUILD_TYPE} BUILD_TYPE)
if (${BUILD_TYPE} STREQUAL debug)
    add_compile_definitions(ENABLE_ASSERT=1)
endif()

set(ZETIC_FEATURE_COMMON_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/inc)

set(ZETIC_FEATURE_COMMON_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/zetic_feature_types.c)

add_library(zetic_feature_common ${LIBRARY_BUILD_TYPE} ${ZETIC_FEATURE_COMMON_SRCS})
target_include_directories(zetic_feature_common PUBLIC ${ZETIC_FEATURE_COMMON_INCLUDE_DIRS})

if(PLATFORM STREQUAL "android")
    target_link_libraries(zetic_feature_common PUBLIC dl android log)
else()
    target_link_libraries(zetic_feature_common PUBLIC dl)
endif()
