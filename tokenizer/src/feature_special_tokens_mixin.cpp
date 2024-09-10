#include "feature_special_tokens_mixin.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>


// TODO: Move to JSON util

// Simple function to extract a string value from JSON-like text
std::string extractStringValue(const std::string& json, const std::string& key) {
    size_t pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return "";
    pos = json.find(":", pos);
    if (pos == std::string::npos) return "";
    pos = json.find("\"", pos);
    if (pos == std::string::npos) return "";
    size_t end = json.find("\"", pos + 1);
    if (end == std::string::npos) return "";
    return json.substr(pos + 1, end - pos - 1);
}

// TODO: Move to JSON util
// Simple function to extract an array of strings from JSON-like text
std::vector<std::string> extractStringArray(const std::string& json, const std::string& key) {
    std::vector<std::string> result;
    size_t pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return result;
    pos = json.find("[", pos);
    if (pos == std::string::npos) return result;
    size_t end = json.find("]", pos);
    if (end == std::string::npos) return result;
    
    std::string array_content = json.substr(pos + 1, end - pos - 1);
    std::stringstream ss(array_content);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item.erase(std::remove(item.begin(), item.end(), '"'), item.end());
        item.erase(std::remove(item.begin(), item.end(), ' '), item.end());
        if (!item.empty()) {
            result.push_back(item);
        }
    }
    return result;
}

void ZeticMLangeSpecialTokensMixin::updateAllSpecialTokens() {
    this->_all_special_tokens.clear();
    if (!_bos_token.empty()) this->_all_special_tokens.insert(_bos_token);
    if (!_eos_token.empty()) this->_all_special_tokens.insert(_eos_token);
    if (!_unk_token.empty()) this->_all_special_tokens.insert(_unk_token);
    if (!_sep_token.empty()) this->_all_special_tokens.insert(_sep_token);
    if (!_pad_token.empty()) this->_all_special_tokens.insert(_pad_token);
    if (!_cls_token.empty()) this->_all_special_tokens.insert(_cls_token);
    if (!_mask_token.empty()) this->_all_special_tokens.insert(_mask_token);
    this->_all_special_tokens.insert(this->_additional_special_tokens.begin(), this->_additional_special_tokens.end());
}

ZeticMLangeSpecialTokensMixin::ZeticMLangeSpecialTokensMixin(const std::string& json_content) {

    this->_bos_token = extractStringValue(json_content, "bos_token");
    this->_eos_token = extractStringValue(json_content, "eos_token");
    this->_unk_token = extractStringValue(json_content, "unk_token");
    this->_sep_token = extractStringValue(json_content, "sep_token");
    this->_pad_token = extractStringValue(json_content, "pad_token");
    this->_cls_token = extractStringValue(json_content, "cls_token");
    this->_mask_token = extractStringValue(json_content, "mask_token");
    this->_additional_special_tokens = extractStringArray(json_content, "additional_special_tokens");

    this->updateAllSpecialTokens();
}

// Method to set a special token
void ZeticMLangeSpecialTokensMixin::set_special_token(const std::string& key, const std::string& value) {
    if (key == "bos_token") this->_bos_token = value;
    else if (key == "eos_token") this->_eos_token = value;
    else if (key == "unk_token") this->_unk_token = value;
    else if (key == "sep_token") this->_sep_token = value;
    else if (key == "pad_token") this->_pad_token = value;
    else if (key == "cls_token") this->_cls_token = value;
    else if (key == "mask_token") this->_mask_token = value;
    else if (key == "additional_special_tokens") {
        this->_additional_special_tokens.push_back(value);
    }
    else {
        throw std::runtime_error("Unknown special token: " + key);
    }
}

std::unordered_map<std::string, TokenValue> ZeticMLangeSpecialTokensMixin::special_tokens_map_extended() const {

        std::unordered_map<std::string, TokenValue> set_attr;
        for (const auto& attr : SPECIAL_TOKENS_ATTRIBUTES) {
            if (attr == "bos_token" && !_bos_token.empty()) set_attr[attr] = TokenValue(_bos_token);
            else if (attr == "eos_token" && !_eos_token.empty()) set_attr[attr] = TokenValue(_eos_token);
            else if (attr == "unk_token" && !_unk_token.empty()) set_attr[attr] = TokenValue(_unk_token);
            else if (attr == "sep_token" && !_sep_token.empty()) set_attr[attr] = TokenValue(_sep_token);
            else if (attr == "pad_token" && !_pad_token.empty()) set_attr[attr] = TokenValue(_pad_token);
            else if (attr == "cls_token" && !_cls_token.empty()) set_attr[attr] = TokenValue(_cls_token);
            else if (attr == "mask_token" && !_mask_token.empty()) set_attr[attr] = TokenValue(_mask_token);
            else if (attr == "additional_special_tokens" && !_additional_special_tokens.empty()) {
                set_attr[attr] = TokenValue(_additional_special_tokens);
            }
        }
        return set_attr;
}



std::unordered_set<std::string> ZeticMLangeSpecialTokensMixin::all_special_tokens_extended() const {
        std::unordered_set<std::string> all_tokens;
        for (const auto& pair : special_tokens_map_extended()) {
            const auto& value = pair.second;
            if (!value.is_vector_type()) {
                all_tokens.insert(value.get_string());
            } else {
                const auto& tokens = value.get_vector();
                all_tokens.insert(tokens.begin(), tokens.end());
            }
        }
        return all_tokens;
}

std::unordered_set<std::string> ZeticMLangeSpecialTokensMixin::all_special_tokens() const {
        return all_special_tokens_extended();
}
