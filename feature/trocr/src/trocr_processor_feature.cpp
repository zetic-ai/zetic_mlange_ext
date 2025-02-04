#include "trocr_processor_feature.h"
#include "dbg_utils.h"

#include <fstream>
#include <random>

#define YOLO8_OUTPUT_DIM1 8400
#define YOLO8_OUTPUT_DIM2 84

// TODO: need to read post processor maybe..
ZeticMLangeTrocrProcessorFeature::ZeticMLangeTrocrProcessorFeature(const char* preprocessor_config_file_path) {
    
    Zetic_MLange_Feature_Result_t ret = ZETIC_MLANGE_FEATURE_FAIL;
    ret = this->readPreprocessorConfigJson(preprocessor_config_file_path);
    if (ret != ZETIC_MLANGE_FEATURE_SUCCESS) {
        ERRLOG("Failed to read preprocessor json file to load TrOCR model: %s", preprocessor_config_file_path);
        return ;
    }
    this->mlange_feature_opencv = new MLangeFeatureOpenCV();
}

ZeticMLangeTrocrProcessorFeature::~ZeticMLangeTrocrProcessorFeature() {
    delete(this->mlange_feature_opencv);
}

// TODO: We assign delete responsibility to user, possible hazard.
Zetic_MLange_Feature_Result_t ZeticMLangeTrocrProcessorFeature::getByteArrayFromImage(cv::Mat &input_img,
                                                                              int8_t *blob) {
    Zetic_MLange_Feature_Result_t ret = ZETIC_MLANGE_FEATURE_FAIL;
    
    ret = this->mlange_feature_opencv->getByteArrayFromImage(input_img, blob);
    if (ret != ZETIC_MLANGE_FEATURE_SUCCESS) {
        ERRLOG("Failed to get float array from image!");
        return ret;
    }

    return ret;
}

Zetic_MLange_Feature_Result_t ZeticMLangeTrocrProcessorFeature::preprocess(cv::Mat& input_img, cv::Mat& output_image) {
    cv::Mat current_mat;
    input_img.convertTo(current_mat, CV_32F);
    if (this->do_resize) {
        cv::resize(current_mat, output_image, this->size);
        current_mat = output_image;
    }
    if (this->do_rescale) {
        output_image = current_mat * this->rescale_factor;
        current_mat = output_image;
    }
    if (this->do_normalize) {
        cv::Mat mean_mat(current_mat.size(), current_mat.type(), this->image_mean);
        cv::Mat std_mat(current_mat.size(), current_mat.type(), this->image_std);
        output_image = (current_mat - mean_mat) / std_mat;
        current_mat = output_image;
    }
    output_image = current_mat;
    return ZETIC_MLANGE_FEATURE_SUCCESS;
}

Zetic_MLange_Feature_Result_t ZeticMLangeTrocrProcessorFeature::postprocess(std::string& output_sentence_result,
                                                                    void* output)
{
    // TODO: need to make it clear that circulation of output ends
    // TODO: check stopping_criteria
    int* tokens = reinterpret_cast<int*>(output);
    for (int i = 0; i < 10; ++i) {
        std::cout << tokens[i] << " ";
    }
    std::cout << std::endl;
    output_sentence_result = "temp output sentence.";
    return ZETIC_MLANGE_FEATURE_SUCCESS;
}

// TODO: change to read frome json file
Zetic_MLange_Feature_Result_t ZeticMLangeTrocrProcessorFeature::readPreprocessorConfigJson(const char* preprocessor_config_file_path) {
    this->do_normalize = true;
    this->do_rescale = true;
    this->do_resize = true;
    this->size = cv::Size(384, 384);
    this->rescale_factor = 0.00392156862745098;
    this->image_mean = cv::Scalar(0.5, 0.5, 0.5);
    this->image_std = cv::Scalar(0.5, 0.5, 0.5);

    return ZETIC_MLANGE_FEATURE_SUCCESS;
}

