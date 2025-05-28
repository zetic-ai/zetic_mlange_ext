#include "jni_utils.h"

std::string convertJStringToCString(JNIEnv *env, jstring j_str) {
    if (j_str == nullptr) {
        return "";
    } else {
        const char *c_str = env->GetStringUTFChars(j_str, nullptr);
        std::string std_str(c_str);
        env->ReleaseStringUTFChars(j_str, c_str);
        return std_str;
    }
}

jstring convertCStringToJString(JNIEnv *env, const std::string &str) {
    if (str.empty()) {
        return nullptr;
    }

    std::vector<jchar> utf16;
    utf16.reserve(str.length());

    const auto *bytes = reinterpret_cast<const unsigned char *>(str.c_str());
    size_t len = str.length();

    for (size_t i = 0; i < len;) {
        uint32_t codepoint = 0;
        unsigned char c = bytes[i];

        if ((c & 0x80) == 0) {
            codepoint = c;
            i += 1;
        } else if ((c & 0xE0) == 0xC0 && i + 1 < len) {
            codepoint = ((c & 0x1F) << 6) | (bytes[i + 1] & 0x3F);
            i += 2;
        } else if ((c & 0xF0) == 0xE0 && i + 2 < len) {
            codepoint = ((c & 0x0F) << 12) |
                        ((bytes[i + 1] & 0x3F) << 6) |
                        (bytes[i + 2] & 0x3F);
            i += 3;
        } else if ((c & 0xF8) == 0xF0 && i + 3 < len) {
            codepoint = ((c & 0x07) << 18) |
                        ((bytes[i + 1] & 0x3F) << 12) |
                        ((bytes[i + 2] & 0x3F) << 6) |
                        (bytes[i + 3] & 0x3F);
            i += 4;

            if (codepoint > 0xFFFF) {
                codepoint -= 0x10000;
                utf16.push_back((jchar) ((codepoint >> 10) | 0xD800));
                utf16.push_back((jchar) ((codepoint & 0x3FF) | 0xDC00));
                continue;
            }
        } else {
            i += 1;
            continue;
        }

        utf16.push_back((jchar) codepoint);
    }

    return env->NewString(utf16.data(), utf16.size());
}

std::vector<float> convertJFloatArrayToCFloatVector(JNIEnv *env, jfloatArray array) {
    if (array == nullptr)
        return {};
    jsize length = env->GetArrayLength(array);
    jfloat *elements = env->GetFloatArrayElements(array, nullptr);
    if (elements == nullptr)
        return {};
    std::vector<float> result(elements, elements + length);
    env->ReleaseFloatArrayElements(array, elements, JNI_ABORT);
    return result;
}

std::vector<int> convertJIntArrayToCIntVector(JNIEnv *env, jintArray array) {
    if (array == nullptr)
        return {};
    jsize length = env->GetArrayLength(array);
    jint *elements = env->GetIntArrayElements(array, nullptr);
    if (elements == nullptr)
        return {};
    std::vector<int> result(elements, elements + length);
    env->ReleaseIntArrayElements(array, elements, JNI_ABORT);
    return result;
}

jfloatArray convertCFloatVectorToJFloatArray(JNIEnv *env, const std::vector<float> &vec) {
    if (vec.empty()) {
        return nullptr;
    }

    jfloatArray result = env->NewFloatArray(vec.size());
    if (result == nullptr) {
        return nullptr;
    }

    env->SetFloatArrayRegion(result, 0, vec.size(), vec.data());

    return result;
}

jintArray convertCIntVectorToJIntArray(JNIEnv *env, const std::vector<int> &vec) {
    if (vec.empty()) {
        return nullptr;
    }

    jintArray result = env->NewIntArray(vec.size());
    if (result == nullptr) {
        return nullptr;
    }

    env->SetIntArrayRegion(result, 0, vec.size(), vec.data());

    return result;
}

jintArray convertCIntArrayToJIntArray(JNIEnv *env, const int* arr, const int& size) {
    if (arr == nullptr)
        return nullptr;

    jintArray result = env->NewIntArray(size);
    if (result == nullptr) {
        return nullptr;
    }

    env->SetIntArrayRegion(result, 0, size, arr);

    return result;
}

jobjectArray convertDoubleCIntVectorToJObjectArray(JNIEnv *env, const std::vector<int> &vec1, const std::vector<int> &vec2) {
    if (vec1.size() != vec2.size()) {
        return nullptr;
    }

    jintArray jVec1Array = convertCIntVectorToJIntArray(env, vec1);
    jintArray jVec2Array = convertCIntVectorToJIntArray(env, vec2);

    if (jVec1Array == nullptr || jVec2Array == nullptr) {
        return nullptr;
    }

    jobjectArray result = env->NewObjectArray(2, env->GetObjectClass(jVec1Array), NULL);

    if (result == nullptr) {
        return nullptr;
    }

    env->SetObjectArrayElement(result, 0, jVec1Array);
    env->SetObjectArrayElement(result, 1, jVec2Array);

    return result;
}

jint convertJEnumToCEnum(JNIEnv *env, const char *c_enum_class_name, jobject j_enum_class_object) {
    jclass enumClass = env->FindClass(c_enum_class_name);
    jmethodID envelopeGetValueMethod = env->GetMethodID(enumClass, "getIndex", "()I");
    jint value = env->CallIntMethod(j_enum_class_object, envelopeGetValueMethod);
    return value;
}

void prepare_buffers(JNIEnv *env,
                     jobjectArray j_input_array,
                     jobjectArray j_output_array,
                     jbyte **input_buf_array,
                     size_t *input_buf_sizes,
                     int32_t given_num_inputs,
                     jbyte **output_buf_array,
                     size_t *output_buf_sizes,
                     int32_t given_num_outputs) {
    jobject byte_object = env->GetObjectArrayElement(j_input_array, 0);

    if (env->IsInstanceOf(byte_object, env->FindClass("java/nio/ByteBuffer"))) {
        prepare_byte_buffer_buffers(env, j_input_array, j_output_array, input_buf_array,
                                    input_buf_sizes, given_num_inputs,
                                    output_buf_array, output_buf_sizes, given_num_outputs);
    } else if (env->IsInstanceOf(byte_object, env->FindClass("[B"))) {
        prepare_byte_array_buffers(env, j_input_array, j_output_array, input_buf_array,
                                   input_buf_sizes, given_num_inputs,
                                   output_buf_array, output_buf_sizes, given_num_outputs);
    }
}

void prepare_byte_array_buffers(JNIEnv *env,
                                jobjectArray j_input_array,
                                jobjectArray j_output_array,
                                jbyte **input_buf_array,
                                size_t *input_buf_sizes,
                                int32_t given_num_inputs,
                                jbyte **output_buf_array,
                                size_t *output_buf_sizes,
                                int32_t given_num_outputs) {
    for (int i = 0; i < given_num_inputs; ++i) {
        auto input_array = (jbyteArray) env->GetObjectArrayElement(j_input_array, i);
        input_buf_sizes[i] = env->GetArrayLength(input_array);
        input_buf_array[i] = env->GetByteArrayElements(input_array, nullptr);
        env->DeleteLocalRef(input_array);
    }

    for (int i = 0; i < given_num_outputs; ++i) {
        auto output_array = (jbyteArray) env->GetObjectArrayElement(j_output_array, i);
        output_buf_sizes[i] = env->GetArrayLength(output_array);
        output_buf_array[i] = env->GetByteArrayElements(output_array, nullptr);
        env->DeleteLocalRef(output_array);
    }
}

void prepare_byte_buffer_buffers(JNIEnv *env,
                                 jobjectArray j_input_array,
                                 jobjectArray j_output_array,
                                 jbyte **input_buf_array,
                                 size_t *input_buf_sizes,
                                 int32_t given_num_inputs,
                                 jbyte **output_buf_array,
                                 size_t *output_buf_sizes,
                                 int32_t given_num_outputs) {
    // Process input ByteBuffers
    for (int i = 0; i < given_num_inputs; ++i) {
        auto input_at = env->GetObjectArrayElement(j_input_array, i);
        input_buf_array[i] = static_cast<jbyte *>(env->GetDirectBufferAddress(input_at));
        input_buf_sizes[i] = env->GetDirectBufferCapacity(input_at);
        env->DeleteLocalRef(input_at);
    }

    // Process output ByteBuffers
    for (int i = 0; i < given_num_outputs; ++i) {
        auto output_array = (jbyteArray) env->GetObjectArrayElement(j_output_array, i);
        output_buf_sizes[i] = env->GetArrayLength(output_array);
        output_buf_array[i] = env->GetByteArrayElements(output_array, nullptr);
        env->DeleteLocalRef(output_array);
    }
}

void dispose_buffers(
        JNIEnv *env,
        jobjectArray j_input_array,
        jobjectArray j_output_array,
        jbyte **input_buf_array,
        int32_t given_num_inputs,
        jbyte **output_buf_array,
        int32_t given_num_outputs) {
    jobject byte_object = env->GetObjectArrayElement(j_input_array, 0);
    if (env->IsInstanceOf(byte_object, env->FindClass("[B"))) {
        for (int i = 0; i < given_num_inputs; ++i) {
            auto input_array = (jbyteArray) env->GetObjectArrayElement(j_input_array, i);
            env->ReleaseByteArrayElements(input_array, input_buf_array[i],
                                          JNI_ABORT);
            env->DeleteLocalRef(input_array);
        }
        for (int i = 0; i < given_num_outputs; ++i) {
            auto output_array = (jbyteArray) env->GetObjectArrayElement(j_output_array, i);
            env->ReleaseByteArrayElements(output_array, output_buf_array[i],
                                          JNI_COMMIT);
            env->DeleteLocalRef(output_array);
        }
    } else {
        for (int i = 0; i < given_num_outputs; ++i) {
            auto output_array = (jbyteArray) env->GetObjectArrayElement(j_output_array, i);
            env->ReleaseByteArrayElements(output_array, output_buf_array[i],
                                          JNI_COMMIT);
            env->DeleteLocalRef(output_array);
        }
    }
}
