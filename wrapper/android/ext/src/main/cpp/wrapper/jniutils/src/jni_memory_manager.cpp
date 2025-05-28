#include "jni_memory_manager.h"

int JNIMemoryManager::default_objects_count = 10;
std::unordered_map<std::string, std::vector<managed_jobject>> JNIMemoryManager::objects;

jobject
JNIMemoryManager::acquire(JNIEnv *env, const char *clazz_name, const char *constructor_signature,
                          ...) {
    va_list args;
    va_start(args, constructor_signature);
    if (objects.find(clazz_name) == objects.end()) {
        va_list init_args;
        va_copy(init_args, args);
        JNIMemoryManager::init(env, clazz_name, constructor_signature, init_args);
        va_end(init_args);
    }

    for (auto &it: objects[clazz_name]) {
        if (it.is_dirty) continue;
        it.is_dirty = true;
        va_list init_args;
        va_copy(init_args, args);
        reset(env, clazz_name, constructor_signature, it.obj, args);
        return it.obj;
    }
    va_list init_args;
    va_copy(init_args, args);
    auto temp_object = create(env, clazz_name, constructor_signature, init_args);
    objects[clazz_name].push_back({temp_object, true});
    va_end(init_args);
    va_end(args);
    return temp_object;
}

void
JNIMemoryManager::clear(const char *clazz_name) {
    for (auto &it: objects[clazz_name]) {
        it.is_dirty = false;
    }
}

void JNIMemoryManager::destroy(JNIEnv *env) {
    for (auto &it: objects) {
        for (auto &obj: it.second) {
            env->DeleteGlobalRef(obj.obj);
        }
    }
}

void JNIMemoryManager::init(JNIEnv *env, const char *clazz_name, const char *constructor_signature,
                            va_list args) {

    std::vector<managed_jobject> temp_objects(default_objects_count);

    std::generate(temp_objects.begin(), temp_objects.end(), [=]() {
        return managed_jobject{
                JNIMemoryManager::create(env, clazz_name, constructor_signature, args), false};
    });

    objects.insert(std::make_pair(clazz_name, temp_objects));
}

jobject
JNIMemoryManager::create(JNIEnv *env, const char *clazz_name, const char *constructor_signature,
                         va_list args) {
    jclass clazz = env->FindClass(clazz_name);
    jmethodID methodID = env->GetMethodID(clazz, "<init>", constructor_signature);
    jobject result = env->NewObjectV(clazz, methodID, args);
    jobject globalObject = env->NewGlobalRef(result);
    env->DeleteLocalRef(result);
    env->DeleteLocalRef(clazz);
    return globalObject;
}

void
JNIMemoryManager::reset(JNIEnv *env, const char *clazz_name, const char *constructor_signature, const jobject& obj, va_list args) {
    jclass clazz = env->FindClass(clazz_name);
    jmethodID resetMethod = env->GetMethodID(clazz, "reset", constructor_signature);
    env->CallVoidMethodV(obj, resetMethod, args);
    if (env->ExceptionOccurred()) {
        env->ExceptionClear();
    }
    env->DeleteLocalRef(clazz);
    va_end(args);
}
