#include <jni.h>

#include "jni_utils.h"
#include "vits_tokenizer.h"

static VitsTokenizer *vits_tokenizer;

extern "C" JNIEXPORT void JNICALL
Java_com_zeticai_mlange_feature_texttospeech_vits_VitsWrapper_nativeInit(
        JNIEnv *env, jobject thiz, jstring vocabulary_path) {
    vits_tokenizer = new VitsTokenizer(convertJStringToCString(env, vocabulary_path));
}

extern "C" JNIEXPORT void JNICALL
Java_com_zeticai_mlange_feature_texttospeech_vits_VitsWrapper_nativeDeinit(
        JNIEnv *env, jobject thiz) {
    delete vits_tokenizer;
}


extern "C" JNIEXPORT jobject JNICALL
Java_com_zeticai_mlange_feature_texttospeech_vits_VitsWrapper_nativeConvertTextToIds(
        JNIEnv *env, jobject thiz, jstring text, jint maxLength) {
    auto ctext = convertJStringToCString(env, text);
    auto [ids, attention_mask] = vits_tokenizer->convertTextToIds(ctext, maxLength);
    auto jIdsArray = convertCIntVectorToJIntArray(env, ids);
    auto jMaskArray = convertCIntVectorToJIntArray(env, attention_mask);

    jclass pairClass = env->FindClass("kotlin/Pair");
    jmethodID pairCtor = env->GetMethodID(pairClass, "<init>",
                                          "(Ljava/lang/Object;Ljava/lang/Object;)V");
    jobject pairObject = env->NewObject(pairClass, pairCtor, jIdsArray, jMaskArray);
    return pairObject;
}