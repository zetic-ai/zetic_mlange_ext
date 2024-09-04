#pragma once

#include "feature_opencv.h"
#include "zetic_feature_types.h"

#include <cmath>
#include <vector>

class Landmark {
public:
    Landmark(float x, float y, float z): x{x}, y{y}, z{z} {}

private:
    float x;
    float y;
    float z;
};

class ZeticMLangeFaceLandmarkFeature {
public:
    ZeticMLangeFaceLandmarkFeature();
    ~ZeticMLangeFaceLandmarkFeature();

    Zetic_MLange_Feature_Result_t preprocess(const cv::Mat& input_img, cv::Mat& input_data);
    Zetic_MLange_Feature_Result_t postprocess(uchar** output_data, std::vector<Landmark>& face_landmark_result);
private:

};