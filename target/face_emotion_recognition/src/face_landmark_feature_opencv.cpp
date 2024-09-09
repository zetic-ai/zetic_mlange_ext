#include "face_landmark_feature_opencv.h"
#include "utils.h"

using namespace ZeticMLange;

ZeticMLangeFaceLandmarkFeature::ZeticMLangeFaceLandmarkFeature() = default;

ZeticMLangeFaceLandmarkFeature::~ZeticMLangeFaceLandmarkFeature() = default;

Zetic_MLange_Feature_Result_t
ZeticMLangeFaceLandmarkFeature::preprocess(const cv::Mat &input_img, const Box &roi,
                                           cv::Mat &input_data) {
    cv::Mat mat;
    cv::resize(input_img, mat, {128, 128});
    cv::Rect rect_roi(roi.xmin, roi.ymin,
                      roi.xmax - roi.xmin,
                      roi.ymax - roi.ymin);
    mat = mat(rect_roi);

    if (*mat.size == 0)
        return ZETIC_MLANGE_FEATURE_FAIL;

    cv::resize(mat, input_data, input_size);
    input_data.convertTo(input_data, CV_32F, 1.f / 255.f, 0);

    return ZETIC_MLANGE_FEATURE_SUCCESS;
}

Zetic_MLange_Feature_Result_t ZeticMLangeFaceLandmarkFeature::postprocess(uchar **output_data,
                                                                          FaceLandmarkResult &face_landmark_result) {
    float *raw_landmarks = reinterpret_cast<float *>(output_data[1]);
    float *raw_confidence = reinterpret_cast<float *>(output_data[0]);

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
