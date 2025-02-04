#include <iostream>

#include <filesystem>

#include "trocr_processor_feature.h"
#include "trocr_generator_feature.h"
#include "ort_model.h"
#include "getopt.h"
#include <opencv4/opencv2/imgcodecs.hpp>

#define TROCR_ENCODER_NUM_MODEL_INPUT 1
#define TROCR_ENCODER_NUM_MODEL_OUTPUT 1
#define TROCR_DECODER_NUM_MODEL_INPUT 2
#define TROCR_DECODER_NUM_MODEL_OUTPUT 1

// TO BE FILLED
Zetic_MLange_Feature_Result_t run_trocr_ort_model(std::string model_file_path,
                    uint8_t** given_input_buffers,
                    int32_t num_given_inputs,
                    uint8_t** given_output_buffers,
                    int32_t num_given_outputs) {

    ort_model::OrtModel* ort_model = new ort_model::OrtModel(model_file_path, false);
    Zetic_MLange_Feature_Result_t ret = ZETIC_MLANGE_FEATURE_FAIL;

    if (!ort_model) {
        printf("Failed to load ort model from path %s", model_file_path.c_str());
        return ZETIC_MLANGE_FEATURE_FAIL;
    }

    // Prepare output tensor
    int8_t num_inputs = 0;
    int8_t num_outputs = 0;
    ret = (Zetic_MLange_Feature_Result_t)ort_model->getIONum(&num_inputs, &num_outputs);
    if (ret != ZETIC_MLANGE_FEATURE_SUCCESS) {
        printf("Failed to run getIONum for OrtModel Profile");
        exit(EXIT_FAILURE);
    }

    size_t* arr_input_tensor_size;
    arr_input_tensor_size = (size_t*)malloc(sizeof(size_t*) * num_inputs);

    size_t* arr_output_tensor_size;
    arr_output_tensor_size = (size_t*)malloc(sizeof(size_t*) * num_outputs);

    ret = (Zetic_MLange_Feature_Result_t)ort_model->getIOSize(arr_input_tensor_size, arr_output_tensor_size);
    if (ret != ZETIC_MLANGE_FEATURE_SUCCESS) {
        printf("Failed to run getIOSize for OrtModel Run");
        exit(EXIT_FAILURE);
    }

    ret = (Zetic_MLange_Feature_Result_t)ort_model->run(given_input_buffers, num_given_inputs, given_output_buffers, num_outputs);
    if (ret != ZETIC_MLANGE_FEATURE_SUCCESS) {
        printf("Failed to run ort model");
        return ret;
    }

    printf("Successed to run ort model!");
    return ret;
}



void test_trocr_ort(std::string encoder_model_file_path, 
                    std::string decoder_model_file_path, 
                    std::string preprocessor_config_file_path, 
                    std::string generator_config_file_path, 
                    std::string img_file_path) {

    cv::Mat img = cv::imread(img_file_path);
    cv::Mat processedImg;

    std::string res;

    ZeticMLangeTrocrProcessorFeature mlangeTrocrProcessorFeature = ZeticMLangeTrocrProcessorFeature(preprocessor_config_file_path.c_str());
    ZeticMLangeTrocrGeneratorFeature mlangeTrocrGeneratorFeature = ZeticMLangeTrocrGeneratorFeature(generator_config_file_path.c_str());

    Zetic_MLange_Feature_Result_t ret = ZETIC_MLANGE_FEATURE_FAIL;
    ret = mlangeTrocrProcessorFeature.preprocess(img, processedImg);
    if (ret != ZETIC_MLANGE_FEATURE_SUCCESS) {
        printf("Failed to preprocess!");
        return ;
    }

    float* blob = new float[processedImg.total() * 3];
    ret = mlangeTrocrProcessorFeature.mlange_feature_opencv->_getFloatArrayFromImage(processedImg, blob);

    // TODO: Implement Running Model
    uint8_t** encoder_input_buffers = (uint8_t**)malloc(sizeof(uint8_t*) * TROCR_ENCODER_NUM_MODEL_INPUT);
    encoder_input_buffers[0] = (uint8_t*)blob;
    
    uint8_t** encoder_output_buffers = (uint8_t**)malloc(sizeof(uint8_t*) * TROCR_ENCODER_NUM_MODEL_OUTPUT);
    run_trocr_ort_model(encoder_model_file_path, 
                        encoder_input_buffers, TROCR_ENCODER_NUM_MODEL_INPUT, 
                        encoder_output_buffers, TROCR_ENCODER_NUM_MODEL_OUTPUT);

    // TODO: should make inital id decoder input
    while (true) {
        // TODO: feat decoder recursively
        break;
    }

    ret = mlangeTrocrProcessorFeature.postprocess(res, (void*)encoder_output_buffers[0]);
    if (ret != ZETIC_MLANGE_FEATURE_SUCCESS) {
        printf("Failed to postprocess!");
        return ;
    }
}




int main(int argc, char **argv) {
    char* encoder_model_file_path = NULL;
    char* decoder_model_file_path = NULL;
    char* input_file_path = NULL;
    char* preprocessor_config_file_path = NULL;
    char* generator_config_file_path = NULL;
    bool enable_nnapi = false;

    int opt;
    while ((opt = getopt(argc, argv, "e:d:i:p:g:hn")) != -1) {
        switch (opt) {
            case 'h':
                printf("Usage: %s -m ORT_MODEL_FILE_PATH"
                       "-c INPUT_COCO_FILE_PATH -i INPUT_IMG_PATH -o OUTPUT_IMG_PATH\n",
                       argv[0]);
                printf("Options:\n");
                printf("  -e ORT Encoder Model File Path\n");
                printf("  -d ORT Decoder Model File Path\n");
                printf("  -i Input image path\n");
                printf("  -p Preprocessor Config File Path\n");
                printf("  -g Generator Config File Path\n");
                printf("  -n [Optional for Using NNAPI] Calculate model with Neural Networks API\n");
                printf("  -h Help messages\n");
                exit(EXIT_SUCCESS);
            case 'e':
                encoder_model_file_path = optarg;
                break;
            case 'd':
                decoder_model_file_path = optarg;
                break;
            case 'i':
                input_file_path = optarg;
                break;
            case 'p':
                preprocessor_config_file_path = optarg;
                break;
            case 'g':
                generator_config_file_path = optarg;
                break;
            case 'n':
                enable_nnapi = true;
                break;
            default:
                printf("Usage: %s -e QNN_ENCODER_MODEL_FILE_PATH -d QNN_DECODER_MODEL_FILE_PATH "
                        "-i INPUT_FILE_PATHS "
                        "-p PREPROCESSOR_CONFIG_FILE_PATHS -g GENERATOR_CONFIG_FILE_PATH [-n]\n",
                        argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    if (!encoder_model_file_path || !decoder_model_file_path || !input_file_path || !preprocessor_config_file_path) {
        printf("ERROR! (Encoder Model / Decoder Model / Config / Input) should be specified\n");
        exit(EXIT_FAILURE);
    }
    
    test_trocr_ort(encoder_model_file_path, decoder_model_file_path, preprocessor_config_file_path, 
                    generator_config_file_path, input_file_path);

    printf("Test TrOCR\n");

    return 0;
}