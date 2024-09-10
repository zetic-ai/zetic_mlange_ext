#pragma once

#include <string>
#include <vector>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <regex>
#include <iostream>
#include <algorithm>

#include <sentencepiece_processor.h>
#include <tokenizers_cpp.h>

#include "feature_token_trie.h"
#include "feature_added_token.h"
#include "feature_special_tokens_mixin.h"

std::string _special_tokens_attributes[] = {
    "bos_token",
    "eos_token",
    "unk_token",
    "sep_token",
    "pad_token",
    "cls_token",
    "mask_token",
    "additional_special_tokens",
};

class ZeticMLangeLlamaTokenizer {
public:
    ZeticMLangeLlamaTokenizer(const std::string& model_path, const std::string& json_path);
    ~ZeticMLangeLlamaTokenizer();

    std::vector<std::string> tokenize(const std::string& text);

    void _llama_tokenize(const std::string& text, std::vector<std::string>* res_tokenized_text);

    std::vector<int32_t> encode(const std::string& input_str, bool bos, bool eos);
    std::string decode(const std::vector<int32_t>& ids);
    size_t getVocabSize();
    
    std::string idToToken(int32_t id);
    int32_t tokenToId(const std::string& token);

private:
    
    void _update_trie(const std::vector<std::string>& unique_no_split_tokens);

    sentencepiece::SentencePieceProcessor sp_model;
    
    ZeticMLangeSpecialTokensMixin* specialTokens;
    
    // TODO: Currently minimind uses legacy
    bool legacy = true;    

    bool split_special_tokens;
    bool do_lower_case;

    std::unordered_set<std::string> all_special_tokens;
    std::unordered_map<std::string, int> _added_tokens_encoder;
    std::unordered_map<int, AddedToken*> _added_tokens_decoder;
    TokensTrie tokens_trie;

    int n_words;
    int bos_id;
    int eos_id;
    int pad_id;
};

