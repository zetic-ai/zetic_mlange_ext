#pragma once

#include "zetic_feature_types.h"
#include "whisper_processor.h"
#include "whisper_tokenizer.h"
#include <string>
#include <vector>

extern "C" {
void nativeInitWhisper(std::string vocabulary_path);
void nativeDeinitWhisper();

float* nativeProcessWhisper(float* audio, int size, int* return_size);
const char* nativeDecodeTokenWhisper(int* ids, int size, bool skip_special_token, int* return_size);
}
