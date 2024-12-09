#include "face_landmark_feature_opencv.h"
#include "nn_utils.h"

using namespace ZeticMLange;

FaceLandmarkFeature::FaceLandmarkFeature() = default;

FaceLandmarkFeature::~FaceLandmarkFeature() = default;

Zetic_MLange_Feature_Result_t
FaceLandmarkFeature::preprocess(const cv::Mat &input_img, const Box &roi,
                                cv::Mat &input_data) {
    if (input_img.empty())
        return ZETIC_MLANGE_FEATURE_FAIL;

    cv::Rect rect_roi(roi.x_min * input_img.cols, roi.y_min * input_img.rows,
                      (roi.x_max - roi.x_min) * input_img.cols,
                      (roi.y_max - roi.y_min) * input_img.rows);
    input_data = input_img(rect_roi);

    cv::resize(input_data, input_data, input_size);
    input_data.convertTo(input_data, CV_32F, 1.f / 255.f, 0);

    return ZETIC_MLANGE_FEATURE_SUCCESS;
}

Zetic_MLange_Feature_Result_t FaceLandmarkFeature::postprocess(uchar **output_data,
                                                               FaceLandmarkResult &face_landmark_result) {
    float *raw_landmarks = reinterpret_cast<float *>(output_data[FACE_LANDMARK_FEATURE_RAW_LANDMARK_OUTPUT_IDX]);
    float *raw_confidence = reinterpret_cast<float *>(output_data[FACE_LANDMARK_FEATURE_RAW_CONFIDENCE_OUTPUT_IDX]);

    face_landmark_result.confidence = sigmoid(raw_confidence[0]);

    if (face_landmark_result.confidence < MIN_SCORE)
        return ZETIC_MLANGE_FEATURE_FAIL;

    for (int i = 0; i < 1404; i += 3) {
        face_landmark_result.landmarks.emplace_back(
                raw_landmarks[i] / input_size.width,
                raw_landmarks[i + 1] / input_size.height,
                raw_landmarks[i + 2] / input_size.width
        );
    }

    return ZETIC_MLANGE_FEATURE_SUCCESS;
}
