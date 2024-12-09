#include "whisper_tokenizer.h"
#include "dbg_util.h"

WhisperTokenizer::WhisperTokenizer(const std::string& vocabulary_path) {
    loadVocabulary(vocabulary_path);
    byte_encoder = bytesToUnicode();
    for (const auto& pair : byte_encoder) {
        byte_decoder[pair.second] = pair.first;
    }
}

std::string WhisperTokenizer::decode(const std::vector<int> &ids, bool skip_special_tokens) {
    std::vector<std::string> text_pieces;

    for (int id: ids) {
        if (skip_special_tokens && special_tokens.find(id) != special_tokens.end()) {
            continue;
        }

        auto it = vocabulary.find(id);
        if (it != vocabulary.end()) {
            text_pieces.push_back(it->second);
        }
    }

    std::string result = mergeTokens(text_pieces);

    return decodeText(result);
}

std::string WhisperTokenizer::mergeTokens(const std::vector<std::string> &tokens) {
    std::string result;

    for (const auto &token: tokens) {
        if (token == "Ġ") {
            result += " ";
        } else {
            result += token;
        }
    }

    size_t start = result.find_first_not_of(" ");
    size_t end = result.find_last_not_of(" ");
    if (start == std::string::npos) return "";

    return result.substr(start, end - start + 1);
}

std::unordered_map<int, std::string> WhisperTokenizer::loadVocabulary(const std::string &file_path) {
    std::ifstream file(file_path);

    if (!file.is_open()) {
        ERRLOG("Unable to open file: %s", file_path.c_str());
        return std::unordered_map<int, std::string>();
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    size_t pos = content.find('{');
    if (pos == std::string::npos) {
        ERRLOG("Invalid JSON format: missing opening brace");
        return std::unordered_map<int, std::string>();
    }

    while (true) {
        size_t key_start = content.find('\"', pos + 1);
        if (key_start == std::string::npos) break;

        size_t key_end = content.find('\"', key_start + 1);
        while (content[key_end - 1] == '\\') {
            if (content[key_end - 2] == '\\')
                break;
            key_end = content.find('\"', key_end + 1);
        }
        if (key_end == std::string::npos) break;

        std::string key = content.substr(key_start + 1, key_end - key_start - 1);

        size_t colon = content.find(':', key_end + 1);
        if (colon == std::string::npos) break;

        size_t value_end = content.find_first_of(",}", colon + 1);
        if (value_end == std::string::npos) break;

        std::string value_str = content.substr(colon + 1, value_end - colon - 1);
        value_str.erase(0, value_str.find_first_not_of(" \n\r\t"));
        value_str.erase(value_str.find_last_not_of(" \n\r\t") + 1);

        try {
            int value = std::stoi(value_str);
            vocabulary[value] = key;
        }
        catch (const std::exception &e) {
            ERRLOG("Invalid number format in JSON: %s", value_str.c_str());
            return std::unordered_map<int, std::string>();
        }

        pos = value_end;
        if (content[value_end] == '}') break;
    }

    if (vocabulary.empty()) {
        ERRLOG("No valid key-value pairs found in JSON");
        return std::unordered_map<int, std::string>();
    }

    return vocabulary;
}

std::map<uint8_t, std::string> WhisperTokenizer::bytesToUnicode() {
    std::vector<uint8_t> bs;
    std::vector<uint32_t> cs;

    for (int i = '!'; i <= '~'; ++i) {
        bs.push_back(static_cast<uint8_t>(i));
        cs.push_back(i);
    }

    for (int i = 0xA1; i <= 0xAC; ++i) {
        bs.push_back(static_cast<uint8_t>(i));
        cs.push_back(i);
    }

    for (int i = 0xAE; i <= 0xFF; ++i) {
        bs.push_back(static_cast<uint8_t>(i));
        cs.push_back(i);
    }

    uint32_t n = 0;
    for (uint16_t b = 0; b < 256; ++b) {
        bool found = false;
        for (uint8_t existing: bs) {
            if (b == existing) {
                found = true;
                break;
            }
        }

        if (!found) {
            bs.push_back(static_cast<uint8_t>(b));
            cs.push_back(256 + n);
            ++n;
        }
    }

    std::map<uint8_t, std::string> result;
    for (size_t i = 0; i < bs.size(); ++i) {
        std::string utf8_str;
        uint32_t code_point = cs[i];

        if (code_point <= 0x7F) {
            utf8_str += static_cast<char>(code_point);
        } else if (code_point <= 0x7FF) {
            utf8_str += static_cast<char>(0xC0 | (code_point >> 6));
            utf8_str += static_cast<char>(0x80 | (code_point & 0x3F));
        } else if (code_point <= 0xFFFF) {
            utf8_str += static_cast<char>(0xE0 | (code_point >> 12));
            utf8_str += static_cast<char>(0x80 | ((code_point >> 6) & 0x3F));
            utf8_str += static_cast<char>(0x80 | (code_point & 0x3F));
        }

        result[bs[i]] = utf8_str;
    }

    return result;
}

std::string WhisperTokenizer::decodeText(const std::string &text) {
    std::vector<uint8_t> bytes;
    bytes.reserve(text.length());

    for (size_t i = 0; i < text.length();) {
        std::string utf8_char;

        if ((text[i] & 0x80) == 0) {
            utf8_char = text.substr(i, 1);
            i += 1;
        } else if ((text[i] & 0xE0) == 0xC0) {
            if (i + 1 < text.length()) {
                utf8_char = text.substr(i, 2);
                i += 2;
            } else {
                ERRLOG("Invalid UTF-8 sequence");
                return "";
            }
        } else if ((text[i] & 0xF0) == 0xE0) {
            if (i + 2 < text.length()) {
                utf8_char = text.substr(i, 3);
                i += 3;
            } else {
                ERRLOG("Invalid UTF-8 sequence");
                return "";
            }
        } else {
            ERRLOG("Invalid UTF-8 sequence");
            return "";
        }

        auto it = byte_decoder.find(utf8_char);
        if (it != byte_decoder.end()) {
            bytes.push_back(it->second);
        } else if (errors == "replace") {
            bytes.push_back('?');
        } else if (errors == "ignore") {
            continue;
        } else {
            ERRLOG("Unknown character in text");
            return "";
        }
    }

    std::string result;
    result.reserve(bytes.size());

    std::string utf8_str(bytes.begin(), bytes.end());

    return utf8_str;
}
