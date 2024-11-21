#pragma once

#include <string>
#include <vector>
#include <cstdio>
#include <opencv2/opencv.hpp>

#include "zetic_feature_types.h"

class MLangeFeatureOpenCV {
public:
    MLangeFeatureOpenCV();
    ~MLangeFeatureOpenCV();

    Zetic_MLange_Feature_Result_t getFloatarrayFromImage(cv::Mat& input_image, float* t_array);
    Zetic_MLange_Feature_Result_t getByteArrayFromImage(cv::Mat& input_image, int8_t* t_array);
    Zetic_MLange_Feature_Result_t getFlatFloatarrayFromImage(cv::Mat& input_image, float* t_array);
        
    Zetic_MLange_Feature_Result_t getLetterBox(cv::Mat& input_img, std::vector<int> input_img_size, cv::Mat& output_image);
    Zetic_MLange_Feature_Result_t getCenterCrop(cv::Mat& input_img, std::vector<int> input_img_size, cv::Mat& output_image);

private:
};
