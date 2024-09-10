
#include <iostream>
#include <string>


#include "feature_llamatokenizer.h"


int main() {

    std::string model_path = "/home/yeonseok/workspace/exp/zetic_mentat_bench/models/minimind/minimind_git/model/tokenizer.model";
    std::string config_path = "/home/yeonseok/workspace/exp/zetic_mentat_bench/models/minimind/minimind_git/model/tokenizer_config.json";


    ZeticMLangeLlamaTokenizer* zeticTokenizer = new ZeticMLangeLlamaTokenizer(model_path, config_path);



    std::string input_text = "<s>user\n给我讲讲硅谷吧</s>\n<s>assistant\n";
    std::vector<std::string> tokenized_text = zeticTokenizer->tokenize(input_text);



    for (int i=0; i<tokenized_text.size(); ++i) {
        printf("%s \n", tokenized_text[i].c_str());
    }

    printf("Test LLama Tokenizer will be updated!\n");



    return 0;
}