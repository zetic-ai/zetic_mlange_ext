#pragma once

#include "feature_opencv.h"
#include "zetic_feature_types.h"
#include "box.h"

namespace ZeticMLange {
    class FaceEmotionRecognitionFeature {
    public:
        FaceEmotionRecognitionFeature();
        ~FaceEmotionRecognitionFeature();

        Zetic_MLange_Feature_Result_t preprocess(const cv::Mat& input_img, const Box& roi, cv::Mat& input_data);
        Zetic_MLange_Feature_Result_t postprocess(uint8_t** output_data, std::pair<float, std::string>& result);

    private:
        // resolution of input image of model
        cv::Size scale = cv::Size(224, 224);
        std::vector<std::string> emotions = {
                "Neutral",
                "Happiness",
                "Sadness",
                "Surprise",
                "Fear",
                "Disgust",
                "Anger"
        };
    };
}
