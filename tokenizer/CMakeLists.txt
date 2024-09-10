
cmake_minimum_required(VERSION 3.18)

project(zetic_mlange_feature_tokenizer C CXX)

# set(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS}")
# set(CMAKE_CUDA_STANDARD 17)

set(MLC_ENABLE_SENTENCEPIECE_TOKENIZER ON)
# set(MSGPACK_USE_BOOST OFF)

set(FEATURE_COMMON_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/inc)

set(TOKENZIER_CPP_PATH ${CMAKE_CURRENT_SOURCE_DIR}/tokenizers-cpp)
add_subdirectory(${TOKENZIER_CPP_PATH} tokenizers EXCLUDE_FROM_ALL)

add_library(zetic_mlange_feature_llamatokenizer SHARED
                            ${CMAKE_CURRENT_SOURCE_DIR}/src/feature_llamatokenizer.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/src/feature_special_tokens_mixin.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/src/feature_token_trie.cpp)

target_include_directories(zetic_mlange_feature_llamatokenizer PUBLIC
                            ${CMAKE_CURRENT_SOURCE_DIR}/inc
                            
                            ${CMAKE_CURRENT_SOURCE_DIR}/tokenizers-cpp/include
                            ${CMAKE_CURRENT_SOURCE_DIR}/tokenizers-cpp/sentencepiece/src
                            ${CMAKE_CURRENT_SOURCE_DIR}/tokenizers-cpp/msgpack/include
                            ${FEATURE_COMMON_INCLUDE_DIRS})

# target_link_libraries(zetic_mlange_feature_llamatokenizer tokenizers_cpp zetic_feature_common)
target_link_libraries(zetic_mlange_feature_llamatokenizer tokenizers_cpp)

add_executable(run_zetic_llama_tokenizer ${CMAKE_CURRENT_SOURCE_DIR}/proc/run_llamatokenizer.cpp)
# add_executable(zetic_qc_runner ${CMAKE_CURRENT_SOURCE_DIR}/proc/qc_runner.cpp)

target_link_libraries(run_zetic_llama_tokenizer PUBLIC zetic_mlange_feature_llamatokenizer)


