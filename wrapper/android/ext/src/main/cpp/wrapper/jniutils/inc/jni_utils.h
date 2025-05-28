#pragma once

#include <jni.h>
#include <string>
#include <vector>

std::string convertJStringToCString(JNIEnv *env, jstring j_str);

jstring convertCStringToJString(JNIEnv *env, const std::string &str);

std::vector<int> convertJIntArrayToCIntVector(JNIEnv *env, jintArray array);

std::vector<float> convertJFloatArrayToCFloatVector(JNIEnv *env, jfloatArray array);

jfloatArray convertCFloatVectorToJFloatArray(JNIEnv *env, const std::vector<float> &vec);

jintArray convertCIntArrayToJIntArray(JNIEnv *env, const int* arr, const int& size);

jintArray convertCIntVectorToJIntArray(JNIEnv *env, const std::vector<int> &vec);

jobjectArray convertDoubleCIntVectorToJObjectArray(JNIEnv *env, const std::vector<int> &vec1, const std::vector<int> &vec2);

jint convertJEnumToCEnum(JNIEnv *env, const char* c_enum_class_name, jobject j_enum_class_object);

void prepare_buffers(JNIEnv *env,
                     jobjectArray j_input_array,
                     jobjectArray j_output_array,
                     jbyte **input_buf_array,
                     size_t *input_buf_sizes,
                     int32_t given_num_inputs,
                     jbyte **output_buf_array,
                     size_t *output_buf_sizes,
                     int32_t given_num_outputs);

void prepare_byte_array_buffers(JNIEnv *env,
                                jobjectArray j_input_array,
                                jobjectArray j_output_array,
                                jbyte **input_buf_array,
                                size_t *input_buf_sizes,
                                int32_t given_num_inputs,
                                jbyte **output_buf_array,
                                size_t *output_buf_sizes,
                                int32_t given_num_outputs);

void prepare_byte_buffer_buffers(JNIEnv *env,
                                 jobjectArray j_input_array,
                                 jobjectArray j_output_array,
                                 jbyte **input_buf_array,
                                 size_t *input_buf_sizes,
                                 int32_t given_num_inputs,
                                 jbyte **output_buf_array,
                                 size_t *output_buf_sizes,
                                 int32_t given_num_outputs);

void dispose_buffers(
        JNIEnv *env,
        jobjectArray j_input_array,
        jobjectArray j_output_array,
        jbyte **input_buf_array,
        int32_t given_num_inputs,
        jbyte **output_buf_array,
        int32_t given_num_outputs);
