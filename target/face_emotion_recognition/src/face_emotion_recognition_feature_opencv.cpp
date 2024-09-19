#include "face_emotion_recognition_feature_opencv.h"

using namespace ZeticMLange;

FaceEmotionRecognition::FaceEmotionRecognition() {

}

FaceEmotionRecognition::~FaceEmotionRecognition() {

}

Zetic_MLange_Feature_Result_t
FaceEmotionRecognition::preprocess(const cv::Mat &input_img,
                                                const Box &roi,
                                                cv::Mat &input_data) {
    if (input_img.empty())
        return ZETIC_MLANGE_FEATURE_FAIL;

    cv::Rect rect_roi(roi.xmin * input_img.cols, roi.ymin * input_img.rows,
                      (roi.xmax - roi.xmin) * input_img.cols,
                      (roi.ymax - roi.ymin) * input_img.rows);
    input_data = input_img(rect_roi);

    cv::resize(input_data, input_data, scale);
    input_data.convertTo(input_data, CV_32F);

    std::vector<cv::Mat> channels(3);
    cv::split(input_data, channels);

    cv::vconcat(channels, input_data);

    return ZETIC_MLANGE_FEATURE_SUCCESS;
}

Zetic_MLange_Feature_Result_t
FaceEmotionRecognition::postprocess(uint8_t **output_data,
                                                 std::pair<float, std::string> &result) {
    float *output = reinterpret_cast<float *>(output_data[0]);
    int argmax = std::distance(output, std::max_element(output, output + 7));

    result = {output[argmax], emotions[argmax]};
    return ZETIC_MLANGE_FEATURE_SUCCESS;
}
