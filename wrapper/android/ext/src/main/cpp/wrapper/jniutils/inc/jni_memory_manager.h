#pragma once

#include "memory_manager.h"
#include <jni.h>
#include <string>
#include <unordered_map>
#include <vector>

struct managed_jobject {
    jobject obj = nullptr;
    bool is_dirty = false;
};

class JNIMemoryManager {
public:
    static jobject
    acquire(JNIEnv *env, const char *clazz_name, const char *constructor_signature, ...);

    static void
    destroy(JNIEnv *env);

    static void
    clear(const char* clazz_name);

    static int default_objects_count;
private:
    static void
    init(JNIEnv *env, const char *clazz_name, const char *constructor_signature, va_list args);

    static jobject
    create(JNIEnv *env, const char *clazz_name, const char *constructor_signature, va_list args);

    static void
    reset(JNIEnv *env, const char *clazz_name, const char *constructor_signature, const jobject& obj, va_list args);

    static std::unordered_map<std::string, std::vector<managed_jobject>> objects;
};
