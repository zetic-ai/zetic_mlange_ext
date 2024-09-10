#include "feature_llamatokenizer.h"
#include "zetic_ext_str_util.h"

#include <string>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <iostream>

#define SPIECE_UNDERLINE "_"

// TODO: REMOVE after enable build all include common: BEGIN

std::string str_trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\r");
    if (std::string::npos == first) {
        return str;
    }
    size_t last = str.find_last_not_of(" \t\n\r");
    return str.substr(first, (last - first + 1));
}

bool str_parse_bool(const std::string& value) {
    return value == "true";
}

std::string str_join(const std::vector<std::string>& v, const std::string& delim) {
    std::string result;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i > 0) result += delim;
        result += v[i];
    }
    return result;
}

std::string str_to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), 
                    [](unsigned char c){ return std::tolower(c); });
    return s;
}

std::string str_trim_left(std::string s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    return s;
}

std::string str_trim_right(std::string s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), s.end());
    return s;
}

// TODO: REMOVE after enable build all include common: END

void ZeticMLangeLlamaTokenizer::_update_trie(const std::vector<std::string>& unique_no_split_tokens = {}) {
    for (const auto& pair : _added_tokens_decoder) {
        const std::string& token = pair.second->getContent();
        if (tokens_trie.tokens.find(token) == tokens_trie.tokens.end()) {
            tokens_trie.add(token);
        }
    }
    for (const auto& token : unique_no_split_tokens) {
        if (tokens_trie.tokens.find(token) == tokens_trie.tokens.end()) {
            tokens_trie.add(token);
        }
    }
}

std::unordered_map<int, AddedToken*> _parse_added_tokens_decoder(const std::string& json_content) {
    std::unordered_map<int, AddedToken*> added_tokens_decoder;
    std::istringstream iss(json_content);
    std::string line;
    int current_id = -1;
    std::string content;
    bool lstrip = false, normalized = false, rstrip = false, single_word = false, special = false;

    while (std::getline(iss, line)) {
        line = str_trim(line);
        if (line == "\"added_tokens_decoder\": {") {
            while (std::getline(iss, line)) {
                line = str_trim(line);
                if (line[0] == '"' && line[line.length() - 1] == '{') {
                    current_id = std::stoi(line.substr(1, line.length() - 4));
                } else if (line.find("\"content\":") != std::string::npos) {
                    content = line.substr(line.find(":") + 2);
                    content = content.substr(1, content.length() - 3);
                } else if (line.find("\"lstrip\":") != std::string::npos) {
                    lstrip = str_parse_bool(line.substr(line.find(":") + 2));
                } else if (line.find("\"rstrip\":") != std::string::npos) {
                    rstrip = str_parse_bool(line.substr(line.find(":") + 2));
                } else if (line.find("\"single_word\":") != std::string::npos) {
                    single_word = str_parse_bool(line.substr(line.find(":") + 2));
                } else if (line.find("\"special\":") != std::string::npos) {
                    special = str_parse_bool(line.substr(line.find(":") + 2));
                } else if (line.find("\"normalized\":") != std::string::npos) {
                    normalized = str_parse_bool(line.substr(line.find(":") + 2));
                } else if ((line == "},") || (line == "}") ) {
                    if (current_id != -1) {
                        added_tokens_decoder[current_id] = new AddedToken(content, lstrip, normalized, rstrip, single_word, special);
                        
                        current_id = -1;
                        content.clear();
                        lstrip = normalized = rstrip = single_word = special = false;
                    }
                }

                if (line == "}") break;
            }
            break;
        }
    }
    return added_tokens_decoder;
}

ZeticMLangeLlamaTokenizer::ZeticMLangeLlamaTokenizer(const std::string& model_path, const std::string& json_path) {

    const auto status = this->sp_model.Load(model_path);
    if (!status.ok()) {
        // TODO: print error
    }

    this->n_words = sp_model.GetPieceSize();
    this->bos_id = sp_model.bos_id();
    this->eos_id = sp_model.eos_id();
    this->pad_id = sp_model.pad_id();


    std::ifstream file(json_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file" << std::endl;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json_content = buffer.str();

    // file is automatically closed when it goes out of scope
    
    this->_added_tokens_decoder = _parse_added_tokens_decoder(json_content);
    this->specialTokens = new ZeticMLangeSpecialTokensMixin(json_content);
    this->all_special_tokens = this->specialTokens->all_special_tokens();
    
    // Create _added_tokens_encoder from _added_tokens_decoder
    for (const auto& pair : _added_tokens_decoder) {
        this->_added_tokens_encoder[pair.second->getContent()] = pair.first;
    }

    _update_trie();

}

ZeticMLangeLlamaTokenizer::~ZeticMLangeLlamaTokenizer() {
    delete(this->specialTokens);
}


void ZeticMLangeLlamaTokenizer::_llama_tokenize(const std::string& text, std::vector<std::string>* res_tokenized_text) {
    this->sp_model.Encode(text, res_tokenized_text);
}

std::vector<std::string> ZeticMLangeLlamaTokenizer::tokenize(const std::string& text) {
        
        // TODO: Currently support !split_special_tokens only
        // bool split_special_tokens = this->split_special_tokens;
        bool split_special_tokens = false;

        // auto it = kwargs.find("split_special_tokens");
        // if (it != kwargs.end()) {
        //     split_special_tokens = (it->second == "true");
        // }

        // auto [prepared_text, remaining_kwargs] = prepare_for_tokenization(text, kwargs);

        auto prepared_text = text;

        // if (!remaining_kwargs.empty()) {
        //     std::cout << "Warning: Keyword arguments not recognized." << std::endl;
        // }

        if (false) {
            std::vector<std::string> escaped_special_toks;
            for (const auto& tok : all_special_tokens) {
                escaped_special_toks.push_back(std::regex_replace(tok, std::regex(R"([.^$*+?()[{\|])"), R"(\$&)"));
            }
            std::string pattern = "(" + str_join(escaped_special_toks, "|") + ")|(.+?)";
            std::regex special_tokens_pattern(pattern);
            
            std::string result;
            std::sregex_iterator it(prepared_text.begin(), prepared_text.end(), special_tokens_pattern);
            std::sregex_iterator end;

            for (; it != end; ++it) {
                if ((*it)[1].length() > 0) {
                    // It's a special token, keep as is
                    result += (*it)[1].str();
                } else {
                    // It's not a special token, convert to lowercase
                    result += str_to_lower((*it)[2].str());
                }
            }

            prepared_text = result;
        }

        std::vector<std::string> tokens;
        
        std::unordered_map<std::string, int> no_split_tokens;
        if (split_special_tokens) {
             no_split_tokens = std::unordered_map<std::string, int>();
            tokens = {prepared_text};
        } else {
            no_split_tokens = _added_tokens_encoder;
            tokens = tokens_trie.split(prepared_text);
        }

        
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (tokens[i].empty()) continue;

            if (_added_tokens_encoder.find(tokens[i]) != _added_tokens_encoder.end()) {
                auto it = _added_tokens_decoder.find(_added_tokens_encoder[tokens[i]]);
                if (it != _added_tokens_decoder.end()) {
                    const AddedToken& tok_extended = *it->second;
                    std::string* left = (i > 0) ? &tokens[i-1] : nullptr;
                    std::string* right = (i < tokens.size() - 1) ? &tokens[i+1] : nullptr;

                    if (tok_extended.isRstrip() && right) {
                        *right = str_trim_left(*right);
                    }
                    if (tok_extended.isLstrip() && left) {
                        *left = str_trim_right(*left);
                    }
                    if (tok_extended.isSingleWord()) {
                        if (left && !left->empty() && left->back() != ' ') {
                            *left += tokens[i];
                            tokens[i].clear();
                        } else if (right && !right->empty() && right->front() != ' ') {
                            *right = tokens[i] + *right;
                            tokens[i].clear();
                        }
                    }
                } else {
                    throw std::runtime_error("Token not properly added to the tokenizer");
                }
                
            } else {
                // auto sub_tokens = _tokenize(tokens[i]);
                // tokenized_text.insert(tokenized_text.end(), sub_tokens.begin(), sub_tokens.end());
            }
        }

        std::vector<std::string> tokenized_text;

        // if (!tokens[i].empty()) {
        //     tokenized_text.push_back(tokens[i]);
        // }

        for (const auto& token : tokens) {
            // Skip empty tokens
            if (token.empty()) {
                continue;
            }

            // Check if the token is in no_split_tokens
            if (no_split_tokens.find(token) != no_split_tokens.end()) {
                tokenized_text.push_back(token);
            } else {
                // Tokenize and extend tokenized_text
                std::vector<std::string> tokenized;
                
                _llama_tokenize(token, &(tokenized));
                tokenized_text.insert(tokenized_text.end(), tokenized.begin(), tokenized.end());
            }
        }


        // Llama tokenize process

        if (legacy) {
            if (tokenized_text.size() > 1 && 
            tokenized_text[0] == SPIECE_UNDERLINE && 
            all_special_tokens.find(tokenized_text[1]) != all_special_tokens.end()) {
            
            // Return a new vector starting from the second element
            return std::vector<std::string>(tokenized_text.begin() + 1, tokenized_text.end());
        }
        }
        
        return tokenized_text;
    }

/*

Encodes a string into a list of token IDs.

Args:
    s (str): The input string to be encoded.
    bos (bool): Whether to prepend the beginning-of-sequence token.
    eos (bool): Whether to append the end-of-sequence token.

Returns:
    std::vector<int32_t>: A list of token IDs.

*/
std::vector<int32_t> ZeticMLangeLlamaTokenizer::encode(const std::string& input_str, bool bos, bool eos) {
    std::vector<int32_t> tokens;
    
    // TODO: Check whether `IgnoreError()` is needed
    this->sp_model.Encode(input_str, &tokens).IgnoreError();

    
    if (bos) {
        tokens.insert(tokens.begin(), bos_id);
    }
    if (eos) {
        tokens.push_back(eos_id);
    }
    return tokens;
}


std::string ZeticMLangeLlamaTokenizer::decode(const std::vector<int32_t>& ids) {
    std::string text;
    this->sp_model.Decode(ids, &text).IgnoreError();
    return text;
}

size_t ZeticMLangeLlamaTokenizer::getVocabSize() {
    auto size = this->sp_model.GetPieceSize();
    return size;
}

std::string ZeticMLangeLlamaTokenizer::idToToken(int32_t id) {
    return this->sp_model.IdToPiece(id);
}


int32_t ZeticMLangeLlamaTokenizer::tokenToId(const std::string& token) {
    return this->sp_model.PieceToId(token);
}

