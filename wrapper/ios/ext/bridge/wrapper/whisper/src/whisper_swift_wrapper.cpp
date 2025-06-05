#include "whisper_swift_wrapper.h"

extern "C" {

WhisperProcessor* whisper_processor;
WhisperTokenizer* whisper_tokenizer;
std::vector<float> process_result;
std::string temp;

void nativeInitWhisper(std::string vocabulary_path) {
    whisper_processor = new WhisperProcessor();
    whisper_tokenizer = new WhisperTokenizer(vocabulary_path);
}

void nativeDeinitWhisper() {
    delete whisper_processor;
    delete whisper_tokenizer;
}

float* nativeProcessWhisper(float* audio, int size, int* return_size) {
    process_result.clear();
    std::vector<float> vector_audio_data;
    vector_audio_data.assign(audio, audio + size);
    process_result = whisper_processor->process(vector_audio_data);
    *return_size = process_result.size();
    return process_result.data();
}

const char* nativeDecodeTokenWhisper(int* ids, int size, bool skip_special_token, int* return_size) {
    std::vector<int> vector_ids;
    vector_ids.assign(ids, ids + size);
    temp = whisper_tokenizer->decode(vector_ids, skip_special_token);
    *return_size = temp.size();
    return temp.c_str();
}
}
