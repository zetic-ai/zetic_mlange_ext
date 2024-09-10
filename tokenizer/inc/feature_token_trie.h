#pragma once

#include <string>
#include <unordered_map>
#include <set>
#include <vector>
#include <algorithm>
#include <iostream>

class TokensTrie {
private:
    struct TrieNode {
        std::unordered_map<char, TrieNode*> children;
        bool is_end;
        
        TrieNode() : is_end(false) {}
        ~TrieNode() {
            for (auto& pair : children) {
                delete pair.second;
            }
        }
    };

    TrieNode* root;
    

public:

    std::set<std::string> tokens;

    TokensTrie() : root(new TrieNode()) {}

    ~TokensTrie() {
        delete root;
    }

    void update(const std::vector<std::string>& words) {
        for (const auto& word : words) {
            add(word);
        }
    }

    void add(const std::string& word) {
        if (word.empty()) {
            return;
        }

        tokens.insert(word);
        TrieNode* node = root;
        for (char c : word) {
            if (node->children.find(c) == node->children.end()) {
                node->children[c] = new TrieNode();
            }
            node = node->children[c];
        }
        node->is_end = true;
    }

    std::vector<std::string> split(const std::string& text) {
        std::vector<int> offsets = {0};
        std::unordered_map<int, TrieNode*> states;
        int skip = 0;

        for (int current = 0; current < text.length(); ++current) {
            if (skip && current < skip) {
                continue;
            }

            std::vector<int> to_remove;
            bool reset = false;

            for (auto& pair : states) {
                int start = pair.first;
                TrieNode* trie_pointer = pair.second;
                int end = current;

                if (trie_pointer->is_end) {
                    for (auto& lookpair : states) {
                        int lookstart = lookpair.first;
                        TrieNode* looktrie_pointer = lookpair.second;

                        if (lookstart > start) {
                            break;
                        }

                        int lookahead_index = (lookstart < start) ? current + 1 : current;
                        end = (lookstart < start) ? current + 1 : current;

                        if (looktrie_pointer->is_end) {
                            start = lookstart;
                            end = lookahead_index;
                            skip = lookahead_index;
                        }

                        while (lookahead_index < text.length() && 
                               looktrie_pointer->children.find(text[lookahead_index]) != looktrie_pointer->children.end()) {
                            looktrie_pointer = looktrie_pointer->children[text[lookahead_index]];
                            ++lookahead_index;
                            if (looktrie_pointer->is_end) {
                                start = lookstart;
                                end = lookahead_index;
                                skip = lookahead_index;
                            }
                        }
                    }

                    offsets.push_back(start);
                    offsets.push_back(end);
                    reset = true;
                    break;
                } else if (trie_pointer->children.find(text[current]) != trie_pointer->children.end()) {
                    pair.second = trie_pointer->children[text[current]];
                } else {
                    to_remove.push_back(start);
                }
            }

            if (reset) {
                states.clear();
            } else {
                for (int start : to_remove) {
                    states.erase(start);
                }
            }

            if (current >= skip && root->children.find(text[current]) != root->children.end()) {
                states[current] = root->children[text[current]];
            }
        }

        for (auto& pair : states) {
            if (pair.second->is_end) {
                offsets.push_back(pair.first);
                offsets.push_back(text.length());
                break;
            }
        }

        return cut_text(text, offsets);
    }

private:
    std::vector<std::string> cut_text(const std::string& text, std::vector<int>& offsets) {
        offsets.push_back(text.length());
        std::sort(offsets.begin(), offsets.end());
        std::vector<std::string> tokens;
        int start = 0;

        for (int end : offsets) {
            if (start > end) {
                std::cerr << "There was a bug in TokensTrie algorithm in tokenization. Attempting to recover. Please report it anyway." << std::endl;
                continue;
            } else if (start == end) {
                continue;
            }
            tokens.push_back(text.substr(start, end - start));
            start = end;
        }

        return tokens;
    }
};