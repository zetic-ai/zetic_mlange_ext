#include "vits_tokenizer.h"
#include "dbg_utils.h"

VitsTokenizer::VitsTokenizer(const std::string &vocabulary_path) {
    loadVocabulary(vocabulary_path);
}

std::pair<std::vector<int>, std::vector<int>>
VitsTokenizer::convertTextToIds(const std::string &text, const int max_length) {
    const std::string normalized_text = normalize_text(text);
    const std::vector<std::string> token_vector =
            intersperseWithId0(normalized_text);
    auto result = convertTokensToIds(token_vector, max_length);
    return result;
}

std::unordered_map<int, std::string>
VitsTokenizer::loadVocabulary(const std::string &file_path) {
    std::ifstream file(file_path);

    if (!file.is_open()) {
        ERRLOG("Unable to open file: %s", file_path.c_str());
        return {};
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    size_t pos = content.find('{');
    if (pos == std::string::npos) {
        ERRLOG("Invalid JSON format: missing opening brace");
        return {};
    }

    while (true) {
        size_t key_start = content.find('\"', pos + 1);
        if (key_start == std::string::npos)
            break;

        size_t key_end = content.find('\"', key_start + 1);
        while (content[key_end - 1] == '\\') {
            if (content[key_end - 2] == '\\')
                break;
            key_end = content.find('\"', key_end + 1);
        }
        if (key_end == std::string::npos)
            break;

        std::string key = content.substr(key_start + 1, key_end - key_start - 1);

        size_t colon = content.find(':', key_end + 1);
        if (colon == std::string::npos)
            break;

        size_t value_end = content.find_first_of(",}", colon + 1);
        if (value_end == std::string::npos)
            break;

        std::string value_str = content.substr(colon + 1, value_end - colon - 1);
        value_str.erase(0, value_str.find_first_not_of(" \n\r\t"));
        value_str.erase(value_str.find_last_not_of(" \n\r\t") + 1);

        try {
            int value = std::stoi(value_str);
            vocabulary[value] = key;
        } catch (const std::exception &e) {
            ERRLOG("Invalid number format in JSON: %s", value_str.c_str());
            return {};
        }

        pos = value_end;
        if (content[value_end] == '}')
            break;
    }

    if (vocabulary.empty()) {
        ERRLOG("No valid key-value pairs found in JSON");
        return {};
    }

    return vocabulary;
}

std::string VitsTokenizer::normalize_text(const std::string &input) {
    std::vector<std::string> all_vocabulary;

    std::string filtered_text;
    filtered_text.reserve(input.size());

    for (unsigned char c: input) {
        filtered_text.push_back(static_cast<char>(std::tolower(c)));
    }

    return filtered_text;
}

std::vector<std::string>
VitsTokenizer::intersperseWithId0(const std::string &text) {

    const std::string pad = "k";

    std::vector<std::string> result;
    result.reserve(text.size() * 2 + 1);

    result.push_back(pad);
    for (unsigned char ch: text) {
        result.emplace_back(1, static_cast<char>(ch));
        result.push_back(pad);
    }
    return result;
}

std::pair<std::vector<int>, std::vector<int>>
VitsTokenizer::convertTokensToIds(const std::vector<std::string> &tokens, const int max_length,
                                  int unk_id, bool throw_on_missing) {
    std::unordered_map<std::string, int> vocab;
    vocab.reserve(vocabulary.size());

    for (const auto &[id, tok]: vocabulary) {
        vocab.emplace(tok, id);
    }

    std::vector<int> ids;
    std::vector<int> attention_mask;

    ids.reserve(max_length);
    attention_mask.reserve(max_length);

    for (const auto &tok: tokens) {
        auto it = vocab.find(tok);
        if (it != vocab.end()) {
            ids.push_back(it->second);
        } else {
            if (throw_on_missing)
                throw std::runtime_error("Unknown token: " + tok);
            ids.push_back(unk_id);
        }
        attention_mask.push_back(1);
    }

    while (ids.size() < static_cast<size_t>(max_length)) {
        ids.push_back(0);
        attention_mask.push_back(0);
    }

    return {ids, attention_mask};
}