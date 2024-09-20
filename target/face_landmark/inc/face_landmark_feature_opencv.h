#pragma once

#include "feature_opencv.h"
#include "zetic_feature_types.h"
#include "data/landmark.h"
#include "data/box.h"
#include "data/face_landmark_result.h"

#include <cmath>
#include <vector>

#define FACE_LANDMARK_FEATURE_RAW_LANDMARK_OUTPUT_IDX 1
#define FACE_LANDMARK_FEATURE_RAW_CONFIDENCE_OUTPUT_IDX 0

namespace ZeticMLange {
    class FaceLandmarkFeature {
    public:
        FaceLandmarkFeature();
        ~FaceLandmarkFeature();

        Zetic_MLange_Feature_Result_t preprocess(const cv::Mat& input_img, const Box& roi, cv::Mat& input_data);
        Zetic_MLange_Feature_Result_t postprocess(uchar** output_data, FaceLandmarkResult& face_landmark_result);
    private:
        // resolution of input image of model
        cv::Size2f input_size = cv::Size2f(192.f, 192.f);
        const float MIN_SCORE = 0.5f;
    };
}
