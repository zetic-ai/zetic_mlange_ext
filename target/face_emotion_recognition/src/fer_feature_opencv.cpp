#include "fer_feature_opencv.h"

ZeticMLangeFERFeature::ZeticMLangeFERFeature() {
    mlange_feature_opencv = new MLangeFeatureOpenCV();
    ssd_generate_anchors();
}

ZeticMLangeFERFeature::~ZeticMLangeFERFeature() {

}

Zetic_MLange_Feature_Result_t ZeticMLangeFERFeature::preprocessFaceDetection(const cv::Mat& input_img, cv::Mat& input_data) {
    if (input_img.empty())
        return ZETIC_MLANGE_FEATURE_FAIL;

    cv::resize(input_img, input_data, cv::Size(128, 128));
    input_data.convertTo(input_data, CV_32F, 1.f / 255.f, 0);

    return ZETIC_MLANGE_FEATURE_SUCCESS;
}

Zetic_MLange_Feature_Result_t ZeticMLangeFERFeature::postprocessFaceDetection(uchar** output_data, std::vector<FaceDetectionResult>& face_detection_results) {
    float* regressors = reinterpret_cast<float*>(output_data[0]);
    float* classificators =  reinterpret_cast<float*>(output_data[1]);

    std::vector<Box> boxes;
    auto result = decodeBoxes(std::vector<float> { regressors, regressors + (896 * 16) }, boxes);

    std::vector<float> scores;
    result = getSigmoidScores(std::vector<float> { classificators, classificators + (896) }, scores);
    if (result != ZETIC_MLANGE_FEATURE_SUCCESS)
        return result;

    std::vector<FaceDetectionResult> detection_result;
    result = convertToDetections(boxes, scores, detection_result);
    if (result != ZETIC_MLANGE_FEATURE_SUCCESS)
        return result;

    return nonMaximumSuppression(detection_result, face_detection_results);
}

Zetic_MLange_Feature_Result_t ZeticMLangeFERFeature::decodeBoxes(const std::vector<float>& raw_boxes, std::vector<Box>& boxes) {
    int num_boxes = raw_boxes.size() / 16;
    int num_points = 8;

    boxes.resize(num_boxes);

    std::vector<std::array<float, 2>> points(num_points);
    for (int i = 0; i < num_boxes; ++i) {
        points.clear();
        for (int j = 0; j < num_points; ++j) {
            points[j] = { raw_boxes[i * num_points * 2 + j * 2] / scale.width,
                          raw_boxes[i * num_points * 2 + j * 2 + 1] / scale.height };
        }

        points[0][0] += anchors[i * 2];
        points[0][1] += anchors[i * 2 + 1];
        for (int j = 2; j < num_points; ++j) {
            points[j][0] += anchors[i * 2];
            points[j][1] += anchors[i * 2 + 1];
        }

        float center_x = points[0][0];
        float center_y = points[0][1];
        float half_size_x = points[1][0] / 2;
        float half_size_y = points[1][1] / 2;

        boxes[i].xmin = center_x - half_size_x;
        boxes[i].ymin = center_y - half_size_y;
        boxes[i].xmax = center_x + half_size_x;
        boxes[i].ymax = center_y + half_size_y;
    }

    return ZETIC_MLANGE_FEATURE_SUCCESS;
}

Zetic_MLange_Feature_Result_t ZeticMLangeFERFeature::getSigmoidScores(const std::vector<float>& raw_scores, std::vector<float>& scores) {
    scores = raw_scores;

    for (size_t i = 0; i < scores.size(); i++) {
        if (scores[i] < -RAW_SCORE_LIMIT)
            scores[i] = -RAW_SCORE_LIMIT;
        else if (scores[i] > RAW_SCORE_LIMIT)
            scores[i] = RAW_SCORE_LIMIT;

        scores[i] = sigmoid(scores[i]);
    }

    return ZETIC_MLANGE_FEATURE_SUCCESS;
}

Zetic_MLange_Feature_Result_t ZeticMLangeFERFeature::convertToDetections(const std::vector<Box>& boxes, const std::vector<float>& scores, std::vector<FaceDetectionResult>& converted_result) {
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (scores[i] > MIN_SCORE && boxes[i].isValid())
            converted_result.push_back(FaceDetectionResult { boxes[i], scores[i] });
    }

    return ZETIC_MLANGE_FEATURE_SUCCESS;
}

Zetic_MLange_Feature_Result_t ZeticMLangeFERFeature::ssd_generate_anchors() {
    int layer_id = 0;
    int num_layers = 4;
    const std::vector<int>& strides = {8, 16, 16, 16};
    if (strides.size() != num_layers)
        return ZETIC_MLANGE_FEATURE_FAIL;
    int input_height = scale.width;
    int input_width = scale.height;
    float anchor_offset_x = 0.5f;
    float anchor_offset_y = 0.5f;
    float interpolated_scale_aspect_ratio = 1.0f;

    while (layer_id < num_layers) {
        int last_same_stride_layer = layer_id;
        int repeats = 0;
        while (last_same_stride_layer < num_layers && strides[last_same_stride_layer] == strides[layer_id]) {
            last_same_stride_layer++;
            repeats += (interpolated_scale_aspect_ratio == 1.0f) ? 2 : 1;
        }

        int stride = strides[layer_id];
        int feature_map_height = input_height / stride;
        int feature_map_width = input_width / stride;

        for (int y = 0; y < feature_map_height; ++y) {
            float y_center = (y + anchor_offset_y) / (float)feature_map_height;
            for (int x = 0; x < feature_map_width; ++x) {
                float x_center = (x + anchor_offset_x) / (float)feature_map_width;
                for (int i = 0; i < repeats; ++i) {
                    anchors.push_back(x_center);
                    anchors.push_back(y_center);
                }
            }
        }

        layer_id = last_same_stride_layer;
    }
    return ZETIC_MLANGE_FEATURE_SUCCESS;
}

Zetic_MLange_Feature_Result_t
ZeticMLangeFERFeature::nonMaximumSuppression(const std::vector<FaceDetectionResult> &detections,
                                             std::vector<FaceDetectionResult> &detections_result) {
    std::vector<float> scores;
    scores.reserve(detections.size());
    for (const auto& detection : detections) {
        scores.push_back(detection.score);
    }

    std::vector<std::pair<int, float>> indexed_scores;
    indexed_scores.reserve(scores.size());
    for (size_t n = 0; n < scores.size(); ++n) {
        indexed_scores.emplace_back(n, scores[n]);
    }

    std::sort(indexed_scores.begin(), indexed_scores.end(),
              [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                  return a.second > b.second;
              });

    auto result = weightedNonMaximumSuppression(indexed_scores, detections, detections_result);
    if (result != ZETIC_MLANGE_FEATURE_SUCCESS)
        return result;

    return ZETIC_MLANGE_FEATURE_SUCCESS;
}

Zetic_MLange_Feature_Result_t ZeticMLangeFERFeature::weightedNonMaximumSuppression(
        const std::vector<std::pair<int, float>>& indexed_scores,
        const std::vector<FaceDetectionResult> &detections,
        std::vector<FaceDetectionResult> &detections_result) {

    std::vector<std::pair<int, float>> remaining_indexed_scores = indexed_scores;
    std::vector<std::pair<int, float>> remaining;
    std::vector<std::pair<int, float>> candidates;

    while (!remaining_indexed_scores.empty()) {
        FaceDetectionResult detection = detections[remaining_indexed_scores[0].first];

        if (detection.score < MIN_SCORE) {
            break;
        }

        size_t num_prev_indexed_scores = remaining_indexed_scores.size();
        Box detection_bbox = detection.bbox;

        remaining.clear();
        candidates.clear();
        FaceDetectionResult weighted_detection = detection;

        for (const auto& pair : remaining_indexed_scores) {
            Box remaining_bbox = detections[pair.first].bbox;
            float similarity = remaining_bbox.overlapSimilarity(detection_bbox);
            if (similarity > MIN_SUPPRESSION_THRESHOLD) {
                candidates.emplace_back(pair.first, pair.second);
            } else {
                remaining.emplace_back(pair.first, pair.second);
            }
        }

        if (!candidates.empty()) {
            Box weighted { 0,0,0,0 };
            float total_score = 0.0f;

            for (const auto& pair : candidates) {
                total_score += pair.second;

                weighted.xmin = detections[pair.first].bbox.xmin * pair.second;
                weighted.ymin = detections[pair.first].bbox.ymin * pair.second;
                weighted.xmax = detections[pair.first].bbox.xmax * pair.second;
                weighted.ymax = detections[pair.first].bbox.ymax * pair.second;
            }

            weighted.xmin = weighted.xmin / total_score;
            weighted.ymin = weighted.ymin / total_score;
            weighted.xmax = weighted.xmax / total_score;
            weighted.ymax = weighted.ymax / total_score;

            weighted_detection = FaceDetectionResult { weighted, detection.score };
        }

        detections_result.push_back(weighted_detection);

        if (num_prev_indexed_scores == remaining.size())
            break;

        remaining_indexed_scores = remaining;
    }

    return ZETIC_MLANGE_FEATURE_SUCCESS;
}
