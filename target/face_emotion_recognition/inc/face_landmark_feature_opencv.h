#pragma once

#include "feature_opencv.h"
#include "zetic_feature_types.h"
#include "landmark.h"
#include "box.h"
#include "face_landmark_result.h"

#include <cmath>
#include <vector>

namespace ZeticMLange {
    class ZeticMLangeFaceLandmarkFeature {
    public:
        ZeticMLangeFaceLandmarkFeature();
        ~ZeticMLangeFaceLandmarkFeature();

        Zetic_MLange_Feature_Result_t preprocess(const cv::Mat& input_img, const Box& roi, cv::Mat& input_data);
        Zetic_MLange_Feature_Result_t postprocess(uchar** output_data, FaceLandmarkResult& face_landmark_result);
    private:
        cv::Size2f input_size = cv::Size2f(192.f, 192.f);
        const float MIN_SCORE = 0.5f;
    };
}
