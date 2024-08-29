#include "fer_feature_opencv.h"

ZeticMLangeFERFeature::ZeticMLangeFERFeature() {
    mlange_feature_opencv = new MLangeFeatureOpenCV();
}

ZeticMLangeFERFeature::~ZeticMLangeFERFeature() {

}

Zetic_MLange_Feature_Result_t ZeticMLangeFERFeature::preprocessFaceDetection(cv::Mat &input_img, float* output_img) {
    cv::Mat resizedFrame;
    cv::resize(input_img, resizedFrame, scale);
    cv::Mat floatFrame;
    resizedFrame.convertTo(floatFrame, CV_32F, 1.0f / 255.0f);

    cv::Mat input_tensor(1, floatFrame.rows * floatFrame.cols * floatFrame.channels(), CV_32F);
    for (int i = 0; i < floatFrame.rows; ++i) {
        for (int j = 0; j < floatFrame.cols; ++j) {
            cv::Vec3f pixel = floatFrame.at<cv::Vec3f>(i, j);
            input_tensor.at<float>(i * floatFrame.cols * floatFrame.channels() + j * floatFrame.channels() + 0) = pixel[0];
            input_tensor.at<float>(i * floatFrame.cols * floatFrame.channels() + j * floatFrame.channels() + 1) = pixel[1];
            input_tensor.at<float>(i * floatFrame.cols * floatFrame.channels() + j * floatFrame.channels() + 2) = pixel[2];
        }
    }

    output_img = (float*)input_tensor.data;
    return ZETIC_MLANGE_FEATURE_SUCCESS;
}

Zetic_MLange_Feature_Result_t ZeticMLangeFERFeature::postprocessFaceDetection(float** output_data, std::vector<FaceDetectionResult>& faceDetectionResults) {
    float* regressors = output_data[0];
    float* classificators = output_data[1];

    std::vector<cv::Rect2f> boxes;
    auto result = decodeBoxes(std::vector<float> { regressors, regressors + (896 * 16 * sizeof(float)) }, boxes);


    std::vector<float> scores;
    result = getSigmoidScores(std::vector<float> { classificators, classificators + (896 * sizeof(float)) }, scores);

    result = convertToDetections(boxes, scores, faceDetectionResults);

    return ZETIC_MLANGE_FEATURE_SUCCESS;
}

Zetic_MLange_Feature_Result_t ZeticMLangeFERFeature::preprocessFaceLandmark(cv::Mat &input_img, cv::Mat &output_img) {
    return ZETIC_MLANGE_FEATURE_FAIL;
}

Zetic_MLange_Feature_Result_t ZeticMLangeFERFeature::preprocessBackbone(cv::Mat &input_img, cv::Mat &output_img) {
    return ZETIC_MLANGE_FEATURE_FAIL;
}

Zetic_MLange_Feature_Result_t ZeticMLangeFERFeature::postprocess(cv::Mat &input_img, cv::Mat &output_img) {
    return ZETIC_MLANGE_FEATURE_FAIL;
}

Zetic_MLange_Feature_Result_t ZeticMLangeFERFeature::decodeBoxes(const std::vector<float>& raw_boxes, std::vector<cv::Rect2f>& boxes) {
    int num_boxes = raw_boxes.size() / (anchors.size() * 2);
    int num_points = anchors.size();
    
    boxes.resize(num_boxes);

    std::vector<cv::Point2f> points(num_boxes * num_points);
    for (int i = 0; i < num_boxes; ++i) {
        for (int j = 0; j < num_points; ++j) {
            points[i * num_points + j] = cv::Point2f(raw_boxes[i * num_points * 2 + j * 2] / scale.width, raw_boxes[i * num_points * 2 + j * 2 + 1] / scale.height);
        }
    }

    for (int i = 0; i < num_boxes; ++i) {
        for (int j = 0; j < num_points; ++j) {
            points[i * num_points + j].x += anchors[j * 2];
            points[i * num_points + j].y += anchors[j * 2 + 1];
        }
    }

    for (int i = 0; i < num_boxes; ++i) {
        cv::Point2f center = points[i * num_points];
        cv::Point2f half_size = points[i * num_points + 1] / 2;
        boxes[i] = cv::Rect2f(center - half_size, center + half_size);
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

Zetic_MLange_Feature_Result_t ZeticMLangeFERFeature::convertToDetections(const std::vector<cv::Rect2f>& boxes, const std::vector<float> scores, std::vector<FaceDetectionResult> converted_result) {
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (scores[i] > MIN_SCORE && isValid(boxes[i]))
            converted_result.push_back(FaceDetectionResult { boxes[i], scores[i] });
    }

    return ZETIC_MLANGE_FEATURE_SUCCESS;
}