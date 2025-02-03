#include <iostream>

#include <filesystem>

#include "yolov8_feature_opencv.h"
#include "ort_model.h"
#include "getopt.h"

#define YOLO_NUM_MODEL_INPUT 1
#define YOLO_NUM_MODEL_OUTPUT 1


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
    ret = ort_model->getIONum(&num_inputs, &num_outputs);
    if (ret != ZETIC_MLANGE_FEATURE_SUCCESS) {
        printf("Failed to run getIONum for OrtModel Profile");
        exit(EXIT_FAILURE);
    }

    size_t* arr_input_tensor_size;
    arr_input_tensor_size = (size_t*)malloc(sizeof(size_t*) * num_inputs);

    size_t* arr_output_tensor_size;
    arr_output_tensor_size = (size_t*)malloc(sizeof(size_t*) * num_outputs);

    ret = ort_model->getIOSize(arr_input_tensor_size, arr_output_tensor_size);
    if (ret != ZETIC_MLANGE_FEATURE_SUCCESS) {
        printf("Failed to run getIOSize for OrtModel Run");
        exit(EXIT_FAILURE);
    }

    ret = ort_model->run(given_input_buffers, num_given_inputs, given_output_buffers, num_outputs);
    if (ret != ZETIC_MLANGE_FEATURE_SUCCESS) {
        printf("Failed to run ort model");
        return ret;
    }

    printf("Successed to run ort model!");
    return ret;
}



void test_trocr_ort(std::string model_file_path, std::string coco_file_path, std::string img_file_path, std::string result_img_file_path, YOLO_MODEL_TYPE yolo_model_type) {

    cv::Mat img = cv::imread(img_file_path);
    cv::Mat processedImg;

    std::vector<DL_RESULT> res;

    MLangeYolo8Feature mlangeYolo8Feature = MLangeYolo8Feature(yolo_model_type, coco_file_path.c_str());

    Zetic_MLange_Feature_Result_t ret = ZETIC_MLANGE_FEATURE_FAIL;
    ret = mlangeYolo8Feature.preprocess(img, processedImg);
    if (ret != ZETIC_MLANGE_FEATURE_SUCCESS) {
        printf("Failed to preprocess!");
        return ;
    }

    float* blob = new float[processedImg.total() * 3];
    ret = mlangeYolo8Feature.getFloatArrayFromImage(processedImg, blob);

    // TODO: Implement Running Model
    uint8_t** input_buffers = (uint8_t**)malloc(sizeof(uint8_t*) * YOLO_NUM_MODEL_INPUT);
    input_buffers[0] = (uint8_t*)blob;
    
    uint8_t** output_buffers = (uint8_t**)malloc(sizeof(uint8_t*) * YOLO_NUM_MODEL_OUTPUT);
    run_trocr_ort_model(model_file_path, 
                        input_buffers, YOLO_NUM_MODEL_INPUT, 
                        output_buffers, YOLO_NUM_MODEL_OUTPUT);
    
    ret = mlangeYolo8Feature.postprocess(res, (void*)output_buffers[0]);
    if (ret != ZETIC_MLANGE_FEATURE_SUCCESS) {
        printf("Failed to postprocess!");
        return ;
    }

    ret = mlangeYolo8Feature.resultToImg(img, res);
    if (ret != ZETIC_MLANGE_FEATURE_SUCCESS) {
        printf("Failed to resultToImg!");
        return ;
    }


    bool result = cv::imwrite(result_img_file_path, img);
    if (!result) {
        std::cout << "Failed to save the image." << std::endl;
        return ;
    }
}




int main(int argc, char **argv) {
    char* model_file_path = NULL;
    char* input_file_paths_arg = NULL;
    char* input_coco_file_paths_arg = NULL;
    char* output_file_paths_arg = NULL;
    bool enable_nnapi = false;

    int opt;
    while ((opt = getopt(argc, argv, "hm:i:c:o:n")) != -1) {
        switch (opt) {
            case 'h':
                printf("Usage: %s -m ORT_MODEL_FILE_PATH"
                       "-c INPUT_COCO_FILE_PATH -i INPUT_IMG_PATH -o OUTPUT_IMG_PATH\n",
                       argv[0]);
                printf("Options:\n");
                printf("  -m ORT Model File Path\n");
                printf("  -c Input Coco File Path\n");
                printf("  -i Input image path\n");
                printf("  -o Output image path\n");
                printf("  -h Help messages\n");
                exit(EXIT_SUCCESS);
            case 'm':
                model_file_path = optarg;
                break;
            case 'i':
                input_file_paths_arg = optarg;
                break;
            case 'c':
                input_coco_file_paths_arg = optarg;
                break;
            case 'o':
                output_file_paths_arg = optarg;
                break;
            case 'n':
                enable_nnapi = true;
                break;
            default:
                printf("Usage: %s -m QNN_MODEL_FILE_PATH -b QNN_BACKEND_PATH"
                        "-i INPUT_FILE_PATHS -o OUTPUT_FILE_PATHS\n",
                        argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    if (!model_file_path || !input_file_paths_arg || !input_coco_file_paths_arg || !output_file_paths_arg) {
        printf("ERROR! (Model / Coco / Input / Output) should be specified\n");
        exit(EXIT_FAILURE);
    }
    
    test_trocr_ort(model_file_path, input_coco_file_paths_arg, input_file_paths_arg, output_file_paths_arg, YOLO_DETECT_V8);

    printf("Test Yolov8\n");

    return 0;
}