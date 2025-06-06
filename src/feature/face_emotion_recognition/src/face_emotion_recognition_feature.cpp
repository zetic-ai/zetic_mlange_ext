#include "face_emotion_recognition_feature.h"

using namespace ZeticMLange;

FaceEmotionRecognitionFeature::FaceEmotionRecognitionFeature() {

}

FaceEmotionRecognitionFeature::~FaceEmotionRecognitionFeature() {

}

Zetic_MLange_Feature_Result_t
FaceEmotionRecognitionFeature::preprocess(const cv::Mat &input_img,
                                                const Box &roi,
                                                cv::Mat &input_data) {
    if (input_img.empty())
        return ZETIC_MLANGE_FEATURE_FAIL;

    cv::Rect rect_roi(roi.x_min * input_img.cols, roi.y_min * input_img.rows,
                      (roi.x_max - roi.x_min) * input_img.cols,
                      (roi.y_max - roi.y_min) * input_img.rows);
    input_data = input_img(rect_roi);

    cv::resize(input_data, input_data, scale);
    input_data.convertTo(input_data, CV_32F);

    std::vector<cv::Mat> channels(3);
    cv::split(input_data, channels);

    cv::vconcat(channels, input_data);

    return ZETIC_MLANGE_FEATURE_SUCCESS;
}

Zetic_MLange_Feature_Result_t
FaceEmotionRecognitionFeature::postprocess(uint8_t **output_data,
                                                 std::pair<float, std::string> &result) {
    float *output = reinterpret_cast<float *>(output_data[0]);
    int argmax = (int)std::distance(output, std::max_element(output, output + 7));

    result = {output[argmax], emotions[argmax]};
    return ZETIC_MLANGE_FEATURE_SUCCESS;
}
