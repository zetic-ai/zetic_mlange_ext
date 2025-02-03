#pragma once

#include <vector>

#include "feature_opencv.h"
#include "zetic_feature_types.h"

class ZeticMLangeTrocrProcessorFeature {
public:
    ZeticMLangeTrocrProcessorFeature(const char* preprocessor_config_file_path);
    ~ZeticMLangeTrocrProcessorFeature();

    Zetic_MLange_Feature_Result_t getByteArrayFromImage(cv::Mat& input_img, int8_t* blob);
    Zetic_MLange_Feature_Result_t preprocess(cv::Mat& input_img, cv::Mat& output_image);
    Zetic_MLange_Feature_Result_t postprocess(std::string& output_sentence_result, void* output);
    
    MLangeFeatureOpenCV* mlange_feature_opencv;
private:
    Zetic_MLange_Feature_Result_t readPreprocessorConfigJson(const char* preprocessor_config_file_path);

    // read from preprocessor config file
    bool do_resize;
    bool do_rescale;
    bool do_normalize;
    // for resize
    cv::Size size;
    // for rescale
    double rescale_factor;
    // for normalize
    cv::Mat image_mean;
    cv::Mat image_std;
};
