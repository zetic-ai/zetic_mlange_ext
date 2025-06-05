#ifndef __ZETIC_FEATURE_TYPES_H__
#define __ZETIC_FEATURE_TYPES_H__

#ifdef __cplusplus
extern "C" {
#endif

typedef enum ZETIC_MLANGE_FEATURE_RESULT {
    ZETIC_MLANGE_FEATURE_SUCCESS = 0,
    ZETIC_MLANGE_FEATURE_FAIL = 1,
    ZETIC_MLANGE_FEATURE_INVALID_INPUT = 2,
    ZETIC_MLANGE_FEATURE_MALLOC_FAIL = 3,
} Zetic_MLange_Feature_Result_t;

#ifdef __cplusplus
}   //  extern "C"
#endif

#endif //   __ZETIC_FEATURE_TYPES_H__
