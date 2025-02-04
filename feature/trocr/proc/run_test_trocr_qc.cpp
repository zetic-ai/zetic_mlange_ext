
#include <iostream>

#include <filesystem>

#include "trocr_processor_feature.h"
#include "trocr_generator_feature.h"
#include "qnn_model.h"
#include "getopt.h"
#include <opencv4/opencv2/imgcodecs.hpp>

#define TROCR_ENCODER_NUM_MODEL_INPUT 1
#define TROCR_ENCODER_NUM_MODEL_OUTPUT 1
#define TROCR_DECODER_NUM_MODEL_INPUT 2
#define TROCR_DECODER_NUM_MODEL_OUTPUT 1

Zetic_MLange_Feature_Result_t run_trocr_qc_model(std::string model_file_path,
                                    std::string backend_file_path,
                                    std::string qnn_graph_name,
                                    uint8_t** given_input_buffers,
                                    int32_t num_given_inputs,
                                    uint8_t** given_output_buffers,
                                    int32_t num_given_outputs) {
    Zetic_MLange_Feature_Result_t ret = ZETIC_MLANGE_FEATURE_FAIL;

    qnn_model_t* qnn_model;
    ret = (Zetic_MLange_Feature_Result_t)qnn_model_init(&qnn_model, (char*)model_file_path.c_str(), (char*)backend_file_path.c_str(), (char*)qnn_graph_name.c_str());
    if (ret != ZETIC_MLANGE_FEATURE_SUCCESS) {
        printf("Failed to init QNN Model!\n");
        exit(EXIT_FAILURE);
    }

    // Prepare output tensor
    int8_t num_inputs = 0;
    int8_t num_outputs = 0;
    ret = (Zetic_MLange_Feature_Result_t)qnn_model_get_io_num(qnn_model, &num_inputs, &num_outputs);
    if (ret != ZETIC_MLANGE_FEATURE_SUCCESS) {
        printf("Failed to run getIONum for QnnModel Profile");
        exit(EXIT_FAILURE);
    }

    size_t* arr_input_num_elems;
    arr_input_num_elems = (size_t*)malloc(sizeof(size_t*) * num_inputs);

    size_t* arr_output_num_elems;
    arr_output_num_elems = (size_t*)malloc(sizeof(size_t*) * num_outputs);

    ret = (Zetic_MLange_Feature_Result_t)qnn_model_get_io_numelems(qnn_model, arr_input_num_elems, arr_output_num_elems);
    if (ret != ZETIC_MLANGE_FEATURE_SUCCESS) {
        printf("Failed to run getIOSize for OrtModel Run");
        exit(EXIT_FAILURE);
    }

    for (int i=0; i<num_outputs; ++i) {
        given_output_buffers[i] = (uint8_t*)malloc(arr_output_num_elems[i] * sizeof(float));
    }

    ret = (Zetic_MLange_Feature_Result_t)qnn_model_run(qnn_model, given_input_buffers, num_given_inputs, given_output_buffers, num_outputs);
    if (ret != ZETIC_MLANGE_FEATURE_SUCCESS) {
        printf("Failed to run ort model");
        return ret;
    }

    printf("Successed to run ort model!");
    return ret;
}



void test_trocr_qnn(std::string encoder_model_file_path, 
                    std::string decoder_model_file_path, 
                    std::string backend_file_path,
                    std::string qnn_graph_name,                
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

    uint8_t** encoder_input_buffers = (uint8_t**)malloc(sizeof(uint8_t*) * TROCR_ENCODER_NUM_MODEL_INPUT);
    encoder_input_buffers[0] = (uint8_t*)blob;

    uint8_t** encoder_output_buffers = (uint8_t**)malloc(sizeof(uint8_t*) * TROCR_ENCODER_NUM_MODEL_OUTPUT);
    run_trocr_qc_model(encoder_model_file_path, 
                        backend_file_path,
                        qnn_graph_name,
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
    char* backend_file_path = NULL;
    char* input_file_path = NULL;
    char* preprocessor_config_file_path = NULL;
    char* generator_config_file_path = NULL;
    char* qnn_graph_name = NULL;

    int opt;
    while ((opt = getopt(argc, argv, "e:d:b:i:p:g:h")) != -1) {
        switch (opt) {
            case 'h':
                printf("Usage: %s -m ORT_MODEL_FILE_PATH"
                       "-c INPUT_COCO_FILE_PATH -i INPUT_IMG_PATH -o OUTPUT_IMG_PATH\n",
                       argv[0]);
                printf("Options:\n");
                printf("  -e QC Encoder Model File Path\n");
                printf("  -d QC Decoder Model File Path\n");
                printf("  -b QC Backend File Path\n");
                printf("  -i Input image path\n");
                printf("  -p Preprocessor Config File Path\n");
                printf("  -g Generator Config File Path\n");
                printf("  -h Help messages\n");
                exit(EXIT_SUCCESS);
            case 'e':
                encoder_model_file_path = optarg;
                break;
            case 'd':
                decoder_model_file_path = optarg;
                break;
            case 'b':
                backend_file_path = optarg;
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
            default:
                printf("Usage: %s -e QNN_ENCODER_MODEL_FILE_PATH -d QNN_DECODER_MODEL_FILE_PATH "
                        "-b QNN_BACKEND_PATH -i INPUT_FILE_PATHS "
                        "-p PREPROCESSOR_CONFIG_FILE_PATHS -g GENERATOR_CONFIG_FILE_PATH\n",
                        argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    if (!encoder_model_file_path ||!backend_file_path ||!decoder_model_file_path 
        || !input_file_path || !preprocessor_config_file_path) {
        printf("ERROR! (Encoder Model / Decoder Model / Config / Input / Backend) should be specified\n");
        exit(EXIT_FAILURE);
    }
    
    printf("Test TrOCR\n");

    test_trocr_qnn(encoder_model_file_path, decoder_model_file_path, 
                    backend_file_path, qnn_graph_name, preprocessor_config_file_path, 
                    generator_config_file_path, input_file_path);

    printf("Test TrOCR End\n");

    return 0;
}