#include <jni.h>

#include "whisper_processor.h"
#include "whisper_tokenizer.h"
#include "jni_utils.h"

static WhisperProcessor* whisper_processor;
static WhisperTokenizer* whisper_tokenizer;

extern "C"
JNIEXPORT void JNICALL
Java_com_zeticai_mlange_feature_whisper_WhisperWrapper_nativeInit(JNIEnv *env, jobject thiz, jstring vocabulary_path) {
    whisper_processor = new WhisperProcessor();
    whisper_tokenizer = new WhisperTokenizer(convertJStringToCString(env, vocabulary_path));
}

extern "C"
JNIEXPORT void JNICALL
Java_com_zeticai_mlange_feature_whisper_WhisperWrapper_nativeDeinit(JNIEnv *env, jobject thiz) {
    delete whisper_processor;
    delete whisper_tokenizer;
}

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_zeticai_mlange_feature_whisper_WhisperWrapper_nativeProcess(JNIEnv *env, jobject thiz,
                                                                     jfloatArray audio) {
    auto vector_audio_data = convertJFloatArrayToCFloatVector(env, audio);
    auto process_result =  whisper_processor->process(vector_audio_data);
    return convertCFloatVectorToJFloatArray(env, process_result);
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_zeticai_mlange_feature_whisper_WhisperWrapper_nativeDecodeToken(JNIEnv *env,
                                                                         jobject thiz,
                                                                         jintArray ids,
                                                                         jboolean skip_special_token) {
    auto vector_ids = convertJIntArrayToCIntVector(env, ids);
    auto token = whisper_tokenizer->decode(vector_ids, skip_special_token);
    return convertCStringToJString(env, token);
}