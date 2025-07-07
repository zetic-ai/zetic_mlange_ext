#pragma once

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <stdexcept>

class VitsTokenizer {

public:
    explicit VitsTokenizer(const std::string &vocabulary_path);

    std::pair<std::vector<int>, std::vector<int>>
    convertTextToIds(const std::string &text, const int max_length);

private:
    std::unordered_map<int, std::string> vocabulary;

    std::unordered_map<int, std::string> loadVocabulary(const std::string &file_path);

    std::string normalize_text(const std::string &input);

    std::vector<std::string> intersperseWithId0(const std::string &text);

    std::pair<std::vector<int>, std::vector<int>>
    convertTokensToIds(const std::vector<std::string> &tokens, const int max_length,
                       int unk_id = -1,
                       bool throw_on_missing = false);
};
