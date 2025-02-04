#pragma once

#include <vector>

#include "feature_opencv.h"
#include "zetic_feature_types.h"

#define VAL_NONE -1

class ZeticMLangeTrocrGeneratorFeature {
public:
    ZeticMLangeTrocrGeneratorFeature(const char* generator_config_file_path);
    ~ZeticMLangeTrocrGeneratorFeature();

    bool stoppingCriteria(std::vector<int> &decoder_input_ids, std::vector<float> &scores);

    // read from generator config file
    // -1 is equal to None
    int max_length = 50;
    int min_length = 0;
    int vocab_size = VAL_NONE;
    int max_position_embeddings = VAL_NONE;
    int bos_token_id = VAL_NONE;
    int start_token_id = VAL_NONE;
    int pad_token_id = VAL_NONE;
    int eos_token_id = VAL_NONE;
private:
    Zetic_MLange_Feature_Result_t readGeneratorConfigJson(const char* generator_config_file_path);
};
