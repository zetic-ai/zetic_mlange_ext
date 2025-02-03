#include "feature_opencv.h"

#include <regex>
#include <opencv2/dnn.hpp>
#define feature_min(a,b) (((a) < (b)) ? (a) : (b))

// TODO: change all MLangeFeatureOpenCV to static

MLangeFeatureOpenCV::MLangeFeatureOpenCV(){}
MLangeFeatureOpenCV::~MLangeFeatureOpenCV(){}

Zetic_MLange_Feature_Result_t MLangeFeatureOpenCV::getFloatArrayFromImage(cv::Mat& input_image, float* t_array) {
    float delimeter_for_division = 1.f / 255.f;
    input_image.convertTo(input_image, CV_32F, delimeter_for_division);

    std::vector<cv::Mat> image_channels;
    cv::split(input_image, image_channels);

    int height = input_image.rows;
    int width = input_image.cols;
    int channels = input_image.channels();

    size_t offset = 0;
    for (int c = 0; c < channels; ++c) {
        std::memcpy(t_array + offset, image_channels[c].data, height * width * sizeof(float));
        offset += height * width;
    }
    return ZETIC_MLANGE_FEATURE_SUCCESS;
}

Zetic_MLange_Feature_Result_t MLangeFeatureOpenCV::getByteArrayFromImage(cv::Mat& input_image, int8_t* t_array) {
    float delimeter_for_division = 1.f / 255.f;
    input_image.convertTo(input_image, CV_32F, delimeter_for_division);

    std::vector<cv::Mat> channel_image;
    cv::split(input_image, channel_image);

    int height = input_image.rows;
    int width = input_image.cols;
    int channels = input_image.channels();

    size_t offset = 0;
    for (int c = 0; c < channels; ++c) {
        std::memcpy(t_array + offset, (int8_t*)channel_image[c].data, height * width * sizeof(float));
        offset += height * width * sizeof(float);
    }

    return ZETIC_MLANGE_FEATURE_SUCCESS;
}

Zetic_MLange_Feature_Result_t MLangeFeatureOpenCV::getFlatFloatArrayFromImage(cv::Mat& input_image, float* t_array) {
    int height = input_image.rows;
    int width = input_image.cols;
    int channels = input_image.channels();

    std::memcpy(t_array, input_image.data, channels * height * width * sizeof(float));
    
    return ZETIC_MLANGE_FEATURE_SUCCESS;
}

Zetic_MLange_Feature_Result_t MLangeFeatureOpenCV::getLetterBox(cv::Mat& input_img, std::vector<int> input_img_size, cv::Mat& output_image) {
    if (input_img.channels() == 3) {
        cv::cvtColor(input_img, output_image, cv::COLOR_BGR2RGB);
    } else {
        cv::cvtColor(input_img, output_image, cv::COLOR_GRAY2RGB);
    }

    if (input_img.cols >= input_img.rows) {
        float resizeScales = input_img.cols / (float)input_img_size.at(0);
        cv::resize(output_image, output_image, cv::Size(int(input_img.cols / resizeScales), input_img_size.at(1)));
    } else {
        float resizeScales = input_img.rows / (float)input_img_size.at(0);
        cv::resize(output_image, output_image, cv::Size(input_img_size.at(0), int(input_img.rows / resizeScales)));
    }
    
    int desired_rows = input_img_size.at(0);
    int desired_cols = input_img_size.at(1);

    int top = 0;
    int left = 0;
    int bottom = desired_rows - output_image.rows;
    int right = desired_cols - output_image.cols;

    if (bottom < 0 || right < 0) {
        return ZETIC_MLANGE_FEATURE_FAIL;
    }

    cv::copyMakeBorder(output_image, output_image, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    return ZETIC_MLANGE_FEATURE_SUCCESS;
}


Zetic_MLange_Feature_Result_t MLangeFeatureOpenCV::getCenterCrop(cv::Mat& input_img, std::vector<int> input_img_size, cv::Mat& output_image) {
    if (input_img.channels() == 3) {
        cv::cvtColor(input_img, output_image, cv::COLOR_BGR2RGB);
    } else {
        cv::cvtColor(input_img, output_image, cv::COLOR_GRAY2RGB);
    }

    int height = input_img.rows;
    int width = input_img.cols;
    int min = feature_min(height, width);
    int top = (height - min) / 2;
    int left = (width - min) / 2;
    cv::resize(output_image(cv::Rect(left, top, min, min)), output_image, cv::Size(input_img_size.at(0), input_img_size.at(1)));
    
    return ZETIC_MLANGE_FEATURE_SUCCESS;
}
