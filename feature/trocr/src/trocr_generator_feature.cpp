#include "trocr_generator_feature.h"
#include "dbg_utils.h"

#include <fstream>

ZeticMLangeTrocrGeneratorFeature::ZeticMLangeTrocrGeneratorFeature(const char* generator_config_file_path) {
    
    Zetic_MLange_Feature_Result_t ret = ZETIC_MLANGE_FEATURE_FAIL;
    ret = this->readGeneratorConfigJson(generator_config_file_path);
    if (ret != ZETIC_MLANGE_FEATURE_SUCCESS) {
        ERRLOG("Failed to read config file to load TrOCR model: %s", generator_config_file_path);
        return ;
    }
}

ZeticMLangeTrocrGeneratorFeature::~ZeticMLangeTrocrGeneratorFeature() {
}

// TODO: change to read frome json file
Zetic_MLange_Feature_Result_t ZeticMLangeTrocrGeneratorFeature::readGeneratorConfigJson(const char* generator_config_file_path) {
    this->max_length = 50;
    this->min_length = 0;
    this->max_position_embeddings = 512;
    this->bos_token_id = 0;
    this->start_token_id = 2;
    this->pad_token_id = 1;
    this->eos_token_id = 2;
}

// TODO: fix it to general input(not only one image output)
bool ZeticMLangeTrocrGeneratorFeature::stoppingCriteria(std::vector<int> &decoder_input_ids, std::vector<float> &scores) {
    size_t cur_len = decoder_input_ids.size();
    bool is_done = cur_len >= max_length;
    if (max_position_embeddings != VAL_NONE && !is_done && cur_len >= max_position_embeddings) {
        DBGLOG("This is a friendly reminder - the current text generation call will exceed the model's predefined "
                "maximum length (%zu). Depending on the model, you may observe "
                "exceptions, performance degradation, or nothing at all.", max_position_embeddings);
    }
    return is_done;
}