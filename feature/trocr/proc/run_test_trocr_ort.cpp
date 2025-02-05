#include <iostream>

#include <filesystem>

#include "trocr_processor_feature.h"
#include "trocr_generator_feature.h"
#include "ort_model.h"
#include "getopt.h"

#define TROCR_ENCODER_NUM_MODEL_INPUT 1
#define TROCR_ENCODER_NUM_MODEL_OUTPUT 1
#define TROCR_DECODER_NUM_MODEL_INPUT 2
#define TROCR_DECODER_NUM_MODEL_OUTPUT 1
// TODO: make use max length is as same as in generator config
// TODO: or change architecture
// TODO: best way is read model and get maximum length of output
#define TROCR_MAX_LENGTH 50

std::string current_path = "";
ort_model::OrtModel* current_model;

Zetic_MLange_Feature_Result_t run_trocr_ort_model(std::string model_file_path,
                    uint8_t** given_input_buffers,
                    int32_t num_given_inputs,
                    uint8_t** given_output_buffers,
                    int32_t num_given_outputs) {

    if (current_path != model_file_path) {
        current_path = model_file_path;
        current_model = new ort_model::OrtModel(model_file_path, false);
    }
    ort_model::OrtModel* ort_model = current_model;
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

    // printf("Successed to run ort model!");
    return ret;
}

// How to run AI model

// (1) Model load
// (2) Setup buffers
// (3+a) Run
// (4) Model deload


// First stage: encoder stage

// (1) encoder load
// (2) encoder setup
// (3) encoder run
// (4) delete(encoder)


// Second stage: decoder stage

// (1) decoder load
// (2) decoder setup

// (3: loop) decoder run

// (4) delete(decoder)


void test_trocr_ort(std::string encoder_model_file_path, 
                    std::string decoder_model_file_path, 
                    std::string preprocessor_config_file_path, 
                    std::string generator_config_file_path, 
                    std::string img_file_path) {
    // load model
    ZeticMLangeTrocrProcessorFeature mlangeTrocrProcessorFeature = ZeticMLangeTrocrProcessorFeature(preprocessor_config_file_path.c_str());
    ZeticMLangeTrocrGeneratorFeature mlangeTrocrGeneratorFeature = ZeticMLangeTrocrGeneratorFeature(generator_config_file_path.c_str());

    int bos_token_id = mlangeTrocrGeneratorFeature.bos_token_id;
    int start_token_id = mlangeTrocrGeneratorFeature.start_token_id;
    int pad_token_id = mlangeTrocrGeneratorFeature.pad_token_id;
    int eos_token_id = mlangeTrocrGeneratorFeature.eos_token_id;
    int vocab_size = mlangeTrocrGeneratorFeature.vocab_size;

    if (start_token_id == VAL_NONE || eos_token_id == VAL_NONE || vocab_size == VAL_NONE) {
        printf("start token or eos token or vocab size is not defined\n");
        return;
    }

    cv::Mat img = cv::imread(img_file_path);
    cv::Mat processedImg;
    std::string res;
    Zetic_MLange_Feature_Result_t ret = ZETIC_MLANGE_FEATURE_FAIL;

    // preprocessing image
    ret = mlangeTrocrProcessorFeature.preprocess(img, processedImg);
    if (ret != ZETIC_MLANGE_FEATURE_SUCCESS) {
        printf("Failed to preprocess!");
        return;
    }

    float* blob = new float[processedImg.total() * 3];
    ret = mlangeTrocrProcessorFeature.mlange_feature_opencv->_getFloatArrayFromImage(processedImg, blob);
    if (ret != ZETIC_MLANGE_FEATURE_SUCCESS) {
        printf("Failed to change image tensor to blob!");
        return ;
    }

    // get result of encoder
    uint8_t** encoder_input_buffers = (uint8_t**)malloc(sizeof(uint8_t*) * TROCR_ENCODER_NUM_MODEL_INPUT);
    encoder_input_buffers[0] = (uint8_t*)blob;
    
    uint8_t** encoder_output_buffers = (uint8_t**)malloc(sizeof(uint8_t*) * TROCR_ENCODER_NUM_MODEL_OUTPUT);
    run_trocr_ort_model(encoder_model_file_path, 
                        encoder_input_buffers, TROCR_ENCODER_NUM_MODEL_INPUT, 
                        encoder_output_buffers, TROCR_ENCODER_NUM_MODEL_OUTPUT);

    // get result of decoder recursivly
    int *decoder_input_ids = new int[TROCR_MAX_LENGTH];
    std::vector<int> decoder_output_ids = {start_token_id, };
    for (int i = 1; i < TROCR_MAX_LENGTH; ++i) {
        decoder_input_ids[i] = pad_token_id;
    }
    decoder_input_ids[0] = start_token_id;

    uint8_t** decoder_input_buffers = (uint8_t**)malloc(sizeof(uint8_t*) * TROCR_DECODER_NUM_MODEL_INPUT);
    uint8_t** decoder_output_buffers = (uint8_t**)malloc(sizeof(uint8_t*) * TROCR_DECODER_NUM_MODEL_OUTPUT);
    decoder_input_buffers[0] = (uint8_t *)decoder_input_ids;
    decoder_input_buffers[1] = encoder_output_buffers[0];

    bool finished = false;
    int current_idx = 0;

    while (true) {
        // TODO: I don't know how allocating output of decoder.
        // TODO: So, I didn't deallocate decoder_output_buffers
        run_trocr_ort_model(decoder_model_file_path,
                            decoder_input_buffers, TROCR_DECODER_NUM_MODEL_INPUT,
                            decoder_output_buffers, TROCR_DECODER_NUM_MODEL_OUTPUT);

        float *decoder_output_scores = (float *)(decoder_output_buffers[0]);

        int max_idx = 0;
        float max_value = decoder_output_scores[vocab_size * current_idx], temp_value;
        for (int i = 1; i < vocab_size; ++i) {
            temp_value = decoder_output_scores[vocab_size * current_idx  + i];
            if (temp_value > max_value) {
                max_idx = i;
                max_value = temp_value;
            }
        }
        decoder_output_ids.push_back(max_idx);
        decoder_input_ids[++current_idx] = max_idx;
        if (max_idx == eos_token_id) {
            finished = true;
        }
        std::vector<float> _score(decoder_output_scores, decoder_output_scores + TROCR_MAX_LENGTH);
        if (mlangeTrocrGeneratorFeature.stoppingCriteria(decoder_output_ids, _score)) {
            finished = true;
        }
        if (finished)
            break;
    }

    for (int id: decoder_output_ids) {
        std::cout << id << ", ";
    }
    std::cout << std::endl;
    std::cout << "token size " << decoder_output_ids.size() << std::endl;
    std::cout << std::endl;

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