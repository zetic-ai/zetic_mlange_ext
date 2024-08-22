#ifndef __ZETIC_DBG_UTIL_H__
#define __ZETIC_DBG_UTIL_H__

#ifdef HAS_BACKTRACE
#include "execinfo.h"
#endif

#include "stdio.h"

#ifdef ANDROID

#include <android/log.h>

#define LOG_TAG "[ZETIC_MLANGE]"
#define DBGLOG(fmt, ...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, fmt, ##__VA_ARGS__)
#define ERRLOG(fmt, ...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, fmt, ##__VA_ARGS__)
#define DASSERT(e)                                  \
    do {                                            \
        if (!(e)) {                                 \
            ERRLOG("ASSERTS(%s) FAILED\n", #e);     \
        }                                           \
    } while(0)

#define DASSERTS(e, format, ...)                                  \
    do {                                            \
        if (!(e)) {                                 \
            ERRLOG("format", ##__VA_ARGS__);        \
        }                                           \
    } while(0)
#else
#define DBGLOG(format, ...)                         \
    do {                                            \
        printf("[ZETIC_MLANGE - DEBUG] %s:%d %s: ",     \
               __FILE__,                            \
               __LINE__,                            \
               __FUNCTION__);                       \
        printf(format, ##__VA_ARGS__);              \
        printf("\n");                               \
    } while(0)

#define ERRLOG(format, ...)                         \
    do {                                            \
        printf("[ZETIC_MLANGE - ERROR] %s:%d %s: ",     \
               __FILE__,                            \
               __LINE__,                            \
               __FUNCTION__);                       \
        printf(format, ##__VA_ARGS__);              \
        printf("\n");                               \
    } while(0)

#ifdef ENABLE_ASSERT

#define DASSERT(e)                                  \
    do {                                            \
        if (!(e)) {                                 \
            ERRLOG("ASSERTS(%s) FAILED\n", #e);     \
            assert(e);                              \
        }                                           \
    } while(0)                                              \

#define DASSERTS(e, format, ...)                                  \
    do {                                            \
        if (!(e)) {                                 \
            ERRLOG("format", ##__VA_ARGS__);        \
            assert(e);                              \
        }                                           \
    } while(0)
#else

#define DASSERT(e)                                  \
    do {                                            \
        if (!(e)) {                                 \
            ERRLOG("ASSERTS(%s) FAILED\n", #e);     \
        }                                           \
    } while(0)

#define DASSERTS(e, format, ...)                                  \
    do {                                            \
        if (!(e)) {                                 \
            ERRLOG("format", ##__VA_ARGS__);        \
        }                                           \
    } while(0)

#endif //   ENABLE_ASSERT

#endif //   ANDROID

#endif //   __ZETIC_DBG_UTIL_H__
