#pragma once

#include "feature_opencv.h"
#include "zetic_feature_types.h"
#include "face_detection_result.h"
#include "nn_utils.h"

#include <vector>

namespace ZeticMLange {
    class FaceDetectionFeature {
    public:
        FaceDetectionFeature();
        ~FaceDetectionFeature();

        Zetic_MLange_Feature_Result_t preprocess(const cv::Mat& input_img, cv::Mat& input_data);
        Zetic_MLange_Feature_Result_t postprocess(uint8_t** output_data, std::vector<FaceDetectionResult>& face_detection_results);

    private:
        Zetic_MLange_Feature_Result_t decodeBoxes(const std::vector<float>& raw_boxes, std::vector<Box>& boxes);
        Zetic_MLange_Feature_Result_t getSigmoidScores(const std::vector<float>& raw_scores, std::vector<float>& scores);
        Zetic_MLange_Feature_Result_t convertToDetections(const std::vector<Box>& boxes, const std::vector<float>& scores, std::vector<FaceDetectionResult>& converted_result);
        Zetic_MLange_Feature_Result_t ssdGenerateAnchors();

        Zetic_MLange_Feature_Result_t nonMaximumSuppression(const std::vector<FaceDetectionResult>& detections, std::vector<FaceDetectionResult>& detections_result);
        Zetic_MLange_Feature_Result_t weightedNonMaximumSuppression(const std::vector<std::pair<int, float>>& indexed_scores, const std::vector<FaceDetectionResult>& detections, std::vector<FaceDetectionResult>& detections_result);

        std::vector<float> anchors;
        // resolution of input image of model
        cv::Size scale = cv::Size(128, 128);

        const int RAW_SCORE_LIMIT = 80;
        const float MIN_SCORE = 0.5f;
        const float MIN_SUPPRESSION_THRESHOLD = 0.3f;

        MLangeFeatureOpenCV* mlange_feature_opencv;
    };

}

