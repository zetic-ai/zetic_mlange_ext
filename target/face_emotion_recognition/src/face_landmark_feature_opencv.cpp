#include "face_landmark_feature_opencv.h"

ZeticMLangeFaceLandmarkFeature::ZeticMLangeFaceLandmarkFeature() = default;

ZeticMLangeFaceLandmarkFeature::~ZeticMLangeFaceLandmarkFeature() = default;

Zetic_MLange_Feature_Result_t
ZeticMLangeFaceLandmarkFeature::preprocess(const cv::Mat &input_img, cv::Mat &input_data) {
    return ZETIC_MLANGE_FEATURE_SUCCESS;
}

Zetic_MLange_Feature_Result_t ZeticMLangeFaceLandmarkFeature::postprocess(uchar **output_data,
                                                                          std::vector<Landmark> &face_landmark_result) {
    return ZETIC_MLANGE_FEATURE_SUCCESS;
}
