#pragma once

#include <string>

class AddedToken {
private:
    std::string content;
    bool special;
    bool lstrip;
    bool rstrip;
    bool single_word;
    bool normalized;

public:
    // AddedToken(const std::string& c, bool l, bool n, bool r, bool sw, bool s)
    //     : content(c), lstrip(l), normalized(n), rstrip(r), single_word(sw), special(s) {}

    AddedToken(const std::string& content, 
               bool special = false, 
               bool lstrip = false, 
               bool rstrip = false, 
               bool single_word = false, 
               bool normalized = true)
        : content(content), 
          special(special), 
          lstrip(lstrip), 
          rstrip(rstrip), 
          single_word(single_word), 
          normalized(normalized) {}

    // Getters
    const std::string& getContent() const { return content; }
    bool isSpecial() const { return special; }
    bool isLstrip() const { return lstrip; }
    bool isRstrip() const { return rstrip; }
    bool isSingleWord() const { return single_word; }
    bool isNormalized() const { return normalized; }

    // Setters
    void setLstrip(bool value) { lstrip = value; }
    void setRstrip(bool value) { rstrip = value; }
    void setSingleWord(bool value) { single_word = value; }

    // Utility methods
    bool shouldStripLeft(const std::string& text_before) const {
        return lstrip && !text_before.empty() && std::isspace(text_before.back());
    }

    bool shouldStripRight(const std::string& text_after) const {
        return rstrip && !text_after.empty() && std::isspace(text_after.front());
    }

    bool shouldSplitBefore(const std::string& text_before) const {
        return !single_word || text_before.empty() || std::isspace(text_before.back());
    }

    bool shouldSplitAfter(const std::string& text_after) const {
        return !single_word || text_after.empty() || std::isspace(text_after.front());
    }
};
