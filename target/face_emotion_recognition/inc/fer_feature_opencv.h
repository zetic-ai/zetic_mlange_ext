#pragma once

#include "feature_opencv.h"
#include "zetic_feature_types.h"

#include <cmath>
#include <vector>

typedef struct _FaceDetectionResult {
    cv::Rect2f bbox;
    float score;
} FaceDetectionResult;

class ZeticMLangeFERFeature {
public:
    ZeticMLangeFERFeature();
    ~ZeticMLangeFERFeature();

    Zetic_MLange_Feature_Result_t preprocessFaceDetection(cv::Mat& input_img, float* input_data);
    Zetic_MLange_Feature_Result_t postprocessFaceDetection(float** output_data, std::vector<FaceDetectionResult>& faceDetectionResults);
    Zetic_MLange_Feature_Result_t preprocessFaceLandmark(cv::Mat& input_img, cv::Mat& output_img);
    Zetic_MLange_Feature_Result_t preprocessBackbone(cv::Mat& input_img, cv::Mat& output_img);
    Zetic_MLange_Feature_Result_t postprocess(cv::Mat& input_img, cv::Mat& output_img);

private:
    float sigmoid(float x) {
        return 1 / (1 + std::exp(-x));
    }

    bool isValid(const cv::Rect2f& bbox) {
        return (bbox.width > 0) && (bbox.height > 0);
    }

    Zetic_MLange_Feature_Result_t decodeBoxes(const std::vector<float>& raw_boxes, std::vector<cv::Rect2f>& boxes);
    Zetic_MLange_Feature_Result_t getSigmoidScores(const std::vector<float>& raw_scores, std::vector<float>& scores);
    Zetic_MLange_Feature_Result_t convertToDetections(const std::vector<cv::Rect2f>& boxes, const std::vector<float> scores, std::vector<FaceDetectionResult> converted_result);

    std::vector<float> anchors;
    cv::Size scale = cv::Size(128, 128);


    const int RAW_SCORE_LIMIT = 80;
    const float MIN_SCORE = 0.5f;
    const float MIN_SUPPRESSION_THRESHOLD = 0.3f;

    MLangeFeatureOpenCV* mlange_feature_opencv;
};
