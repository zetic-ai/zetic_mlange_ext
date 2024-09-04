#pragma once

#include "feature_opencv.h"
#include "zetic_feature_types.h"

#include <cmath>
#include <vector>

class Box {
public:
    Box() : Box(0, 0, 0, 0) {}
    Box(float xmin, float ymin, float xmax, float ymax) : xmin{xmin}, ymin{ymin}, xmax{xmax}, ymax{ymax} {}
    Box(const cv::Rect2f rect) : xmin{rect.x - (rect.width / 2)}, ymin{rect.y - (rect.height / 2)}, xmax{rect.x + (rect.width / 2)}, ymax{rect.y + (rect.height / 2)} {}
    Box(const Box& other) : xmin{other.xmin}, ymin {other.ymin}, xmax {other.xmax}, ymax{other.ymax} {}

    bool isValid() const {
        return (xmin < xmax) && (ymin < ymax);
    }

    Box intersect(const Box& other) const {
        if (xmin < xmax && ymin < ymax) {
            return Box(xmin, ymin, xmax - xmin, ymax - ymin);
        } else {
            return Box(0,0,0,0);
        }
    }

    float area() const {
        return (xmax - xmin) * (ymax - ymin);
    }

    float overlapSimilarity(const Box& other) {
        Box intersection = intersect(other);
        if (!intersection.isValid())
            return 0.0f;
        float intersection_area = intersection.area();
        float denominator = (area()) + (other.area()) - intersection_area;
        if (denominator > 0.0f) {
            return intersection_area / denominator;
        } else {
            return 0.0f;
        }
    }

    float xmin;
    float ymin;
    float xmax;
    float ymax;
};

class FaceDetectionResult {
public:
    Box bbox;
    float score;
};

class ZeticMLangeFERFeature {
public:
    ZeticMLangeFERFeature();
    ~ZeticMLangeFERFeature();

    Zetic_MLange_Feature_Result_t preprocessFaceDetection(const cv::Mat& input_img, cv::Mat& input_data);
    Zetic_MLange_Feature_Result_t postprocessFaceDetection(uchar** output_data, std::vector<FaceDetectionResult>& face_detection_results);

private:
    float sigmoid(float x) {
        return 1.f / (1.f + std::exp(-x));
    }

    Zetic_MLange_Feature_Result_t decodeBoxes(const std::vector<float>& raw_boxes, std::vector<Box>& boxes);
    Zetic_MLange_Feature_Result_t getSigmoidScores(const std::vector<float>& raw_scores, std::vector<float>& scores);
    Zetic_MLange_Feature_Result_t convertToDetections(const std::vector<Box>& boxes, const std::vector<float>& scores, std::vector<FaceDetectionResult>& converted_result);
    Zetic_MLange_Feature_Result_t ssd_generate_anchors();

    Zetic_MLange_Feature_Result_t nonMaximumSuppression(const std::vector<FaceDetectionResult>& detections, std::vector<FaceDetectionResult>& detections_result);
    Zetic_MLange_Feature_Result_t weightedNonMaximumSuppression(const std::vector<std::pair<int, float>>& indexed_scores, const std::vector<FaceDetectionResult>& detections, std::vector<FaceDetectionResult>& detections_result);

    std::vector<float> anchors;
    cv::Size scale = cv::Size(128, 128);

    const int RAW_SCORE_LIMIT = 80;
    const float MIN_SCORE = 0.5f;
    const float MIN_SUPPRESSION_THRESHOLD = 0.3f;

    MLangeFeatureOpenCV* mlange_feature_opencv;
};
