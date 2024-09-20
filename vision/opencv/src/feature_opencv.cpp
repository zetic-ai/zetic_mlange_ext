#include "feature_opencv.h"

#include <regex>

#define feature_min(a,b) (((a) < (b)) ? (a) : (b))

MLangeFeatureOpenCV::MLangeFeatureOpenCV(){}
MLangeFeatureOpenCV::~MLangeFeatureOpenCV(){}


Zetic_MLange_Feature_Result_t MLangeFeatureOpenCV::getFloatarrayFromImage(cv::Mat& input_image, float* t_array) {
    int channels = input_image.channels();
    int imgHeight = input_image.rows;
    int imgWidth = input_image.cols;

    for (int c = 0; c < channels; c++)
    {
        for (int h = 0; h < imgHeight; h++)
        {
            for (int w = 0; w < imgWidth; w++)
            {
                t_array[c * imgWidth * imgHeight + h * imgWidth + w] = (input_image.at<cv::Vec3b>(h, w)[c]) / 255.0f;
            }
        }
    }
    return ZETIC_MLANGE_FEATURE_SUCCESS;
}

Zetic_MLange_Feature_Result_t MLangeFeatureOpenCV::getFlatFloatarrayFromImage(cv::Mat& input_image, float* t_array) {
    int channels = input_image.channels();
    int imgHeight = input_image.rows;
    int imgWidth = input_image.cols;

    std::memcpy(t_array, input_image.data, channels * imgHeight * imgWidth * sizeof(float));
    
    return ZETIC_MLANGE_FEATURE_SUCCESS;
}

Zetic_MLange_Feature_Result_t MLangeFeatureOpenCV::getLetterBox(cv::Mat& input_img, std::vector<int> input_img_size, cv::Mat& output_image) {
    if (input_img.channels() == 3) {
        output_image = input_img.clone();
        cv::cvtColor(output_image, output_image, cv::COLOR_BGR2RGB);
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
    cv::Mat tempImg = cv::Mat::zeros(input_img_size.at(0), input_img_size.at(1), CV_8UC3);
    output_image.copyTo(tempImg(cv::Rect(0, 0, output_image.cols, output_image.rows)));
    output_image = tempImg;

    return ZETIC_MLANGE_FEATURE_SUCCESS;
}


Zetic_MLange_Feature_Result_t MLangeFeatureOpenCV::getCenterCrop(cv::Mat& input_img, std::vector<int> input_img_size, cv::Mat& output_image) {
    if (input_img.channels() == 3) {
        output_image = input_img.clone();
        cv::cvtColor(output_image, output_image, cv::COLOR_BGR2RGB);
    } else {
        cv::cvtColor(input_img, output_image, cv::COLOR_GRAY2RGB);
    }

    int h = input_img.rows;
        int w = input_img.cols;
        int m = feature_min(h, w);
        int top = (h - m) / 2;
        int left = (w - m) / 2;
        cv::resize(output_image(cv::Rect(left, top, m, m)), output_image, cv::Size(input_img_size.at(0), input_img_size.at(1)));

    return ZETIC_MLANGE_FEATURE_SUCCESS;
}
