#pragma once

#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <memory>

class TokenValue {
    public:
        TokenValue() : is_vector(false) {}
        TokenValue(const std::string& s) : str_value(s), is_vector(false) {}
        TokenValue(const std::vector<std::string>& v) : vec_value(v), is_vector(true) {}

        bool is_vector_type() const { return is_vector; }
        const std::string& get_string() const { return str_value; }
        const std::vector<std::string>& get_vector() const { return vec_value; }

    private:
        std::string str_value;
        std::vector<std::string> vec_value;
        bool is_vector;
};

class ZeticMLangeSpecialTokensMixin {
private:
    const std::vector<std::string> SPECIAL_TOKENS_ATTRIBUTES = {
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
        "additional_special_tokens"
    };

    // Private member variables for special tokens
    std::string _bos_token;
    std::string _eos_token;
    std::string _unk_token;
    std::string _sep_token;
    std::string _pad_token;
    std::string _cls_token;
    std::string _mask_token;
    std::vector<std::string> _additional_special_tokens;
    std::unordered_set<std::string> _all_special_tokens;
    int _pad_token_type_id;

    void updateAllSpecialTokens();

public:
    ZeticMLangeSpecialTokensMixin(const std::string& json_content);

    void set_special_token(const std::string& key, const std::string& value);

    std::unordered_map<std::string, TokenValue> special_tokens_map_extended() const;

    std::unordered_set<std::string> all_special_tokens_extended() const;

    std::unordered_set<std::string> all_special_tokens() const;

    const std::unordered_set<std::string>& all_special_tokens_set() const;
};