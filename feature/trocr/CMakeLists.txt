cmake_minimum_required(VERSION 3.13)

project(trocr_feature)

add_library(trocr_feature ${LIBRARY_BUILD_TYPE}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/trocr_processor_feature.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/trocr_generator_feature.cpp
)

target_include_directories(trocr_feature PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/inc)

target_include_directories(trocr_feature PUBLIC ${OpenCV_INCLUDE_DIRS})

if (PLATFORM STREQUAL "android")
    target_link_libraries(trocr_feature
            ${OpenCV_LIBS}
            opencv_feature
            common_feature
            m
            android
            log)

elseif (PLATFORM STREQUAL "ios")
    target_link_libraries(trocr_feature
            ${OpenCV_LIBS}
            opencv_feature
            common_feature)
else ()
    target_link_libraries(trocr_feature
            ${OpenCV_LIBS}
            opencv_feature
            common_feature)
endif ()

# TODO: Test executables need mlange_ort, mlange_qc. 
add_executable(run_test_trocr_qc
            ${CMAKE_CURRENT_SOURCE_DIR}/proc/run_test_trocr_qc.cpp)

target_link_libraries(run_test_trocr_qc trocr_feature)
target_link_libraries(run_test_trocr_qc zetic_mlange_qnn)


# add_executable(run_test_trocr_ort
#                 ${CMAKE_CURRENT_SOURCE_DIR}/proc/run_test_trocr_ort.cpp)
# target_link_libraries(run_test_trocr_ort trocr_feature)
# target_link_libraries(run_test_trocr_ort zetic_mlange_qnn)
