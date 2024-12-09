#pragma once

#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <codecvt>
#include <locale>
#include <cstdint>
#include <map>

class WhisperTokenizer {
public:
    explicit WhisperTokenizer(const std::string& vocabulary_path) {
        loadVocabulary(vocabulary_path);
        byte_encoder = bytesToUnicode();
        for (const auto& pair : byte_encoder) {
            byte_decoder[pair.second] = pair.first;
        }
    }

    std::string decode(const std::vector<int>& ids, bool skip_special_tokens = true);

private:
    std::string mergeTokens(const std::vector<std::string>& tokens);
    std::unordered_map<int, std::string> loadVocabulary(const std::string& filepath);
    std::map<uint8_t, std::string> bytesToUnicode();
    std::string decodeText(const std::string& text);

    std::unordered_set<int> special_tokens;
    std::unordered_map<int, std::string> vocabulary;
    std::map<uint8_t, std::string> byte_encoder;
    std::map<std::string, uint8_t> byte_decoder;
    std::string errors;
};