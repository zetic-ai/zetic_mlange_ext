#include <iostream>

#include <filesystem>

#include "yolo8_feature_opencv.h"
#include "qnn_model.h"
#include "getopt.h"

#define YOLO_NUM_MODEL_INPUT 1
#define YOLO_NUM_MODEL_OUTPUT 1

Zetic_MLange_Feature_Result_t run_yolo_qc_model(std::string model_file_path,
                                    std::string backend_file_path,
                                    std::string qnn_graph_name,
                                    uint8_t** given_input_buffers,
                                    int32_t num_given_inputs,
                                    uint8_t** given_output_buffers,
                                    int32_t num_given_outputs) {
    Zetic_MLange_Feature_Result_t ret = ZETIC_MLANGE_FEATURE_FAIL;

    qnn_model_t* qnn_model;
    ret = qnn_model_init(&qnn_model, (char*)model_file_path.c_str(), (char*)backend_file_path.c_str(), (char*)qnn_graph_name.c_str());
    if (ret != ZETIC_MLANGE_FEATURE_SUCCESS) {
        printf("Failed to init QNN Model!\n");
        exit(EXIT_FAILURE);
    }

    // Prepare output tensor
    int8_t num_inputs = 0;
    int8_t num_outputs = 0;
    ret = qnn_model_get_io_num(qnn_model, &num_inputs, &num_outputs);
    if (ret != ZETIC_MLANGE_FEATURE_SUCCESS) {
        printf("Failed to run getIONum for QnnModel Profile");
        exit(EXIT_FAILURE);
    }

    size_t* arr_input_num_elems;
    arr_input_num_elems = (size_t*)malloc(sizeof(size_t*) * num_inputs);

    size_t* arr_output_num_elems;
    arr_output_num_elems = (size_t*)malloc(sizeof(size_t*) * num_outputs);

    ret = qnn_model_get_io_numelems(qnn_model, arr_input_num_elems, arr_output_num_elems);
    if (ret != ZETIC_MLANGE_FEATURE_SUCCESS) {
        printf("Failed to run getIOSize for OrtModel Run");
        exit(EXIT_FAILURE);
    }

    for (int i=0; i<num_outputs; ++i) {
        given_output_buffers[i] = (uint8_t*)malloc(arr_output_num_elems[i] * sizeof(float));
    }

    ret = qnn_model_run(qnn_model, given_input_buffers, num_given_inputs, given_output_buffers, num_outputs);
    if (ret != ZETIC_MLANGE_FEATURE_SUCCESS) {
        printf("Failed to run ort model");
        return ret;
    }

    printf("Successed to run ort model!");
    return ret;
}



void test_yolo8_qnn(std::string model_file_path, 
                    std::string backend_file_path,
                    std::string qnn_graph_name,                
                    std::string coco_file_path, 
                    std::string img_file_path, 
                    std::string result_img_file_path, 
                    YOLO_MODEL_TYPE yolo_model_type) {

    cv::Mat img = cv::imread(img_file_path);
    cv::Mat processedImg;

    std::vector<DL_RESULT> res;

    ZeticMLangeYoloV8Feature mlangeYoloV8Feature = ZeticMLangeYoloV8Feature(yolo_model_type, coco_file_path.c_str());

    Zetic_MLange_Feature_Result_t ret = ZETIC_MLANGE_FEATURE_FAIL;
    ret = mlangeYoloV8Feature.preprocess(img, processedImg);
    if (ret != ZETIC_MLANGE_FEATURE_SUCCESS) {
        printf("Failed to preprocess!");
        return ;
    }

    float* blob = new float[processedImg.total() * 3];
    ret = mlangeYoloV8Feature.getFloatArrayFromImage(processedImg, blob);

    uint8_t** input_buffers = (uint8_t**)malloc(sizeof(uint8_t*) * YOLO_NUM_MODEL_INPUT);
    input_buffers[0] = (uint8_t*)blob;

    uint8_t** output_buffers = (uint8_t**)malloc(sizeof(uint8_t*) * YOLO_NUM_MODEL_OUTPUT);
    run_yolo_qc_model(model_file_path, 
                        backend_file_path,
                        qnn_graph_name,
                        input_buffers, YOLO_NUM_MODEL_INPUT, 
                        output_buffers, YOLO_NUM_MODEL_OUTPUT);
    
    ret = mlangeYolo8Feature.postprocess(res, (void*)output_buffers[0]);
    if (ret != ZETIC_MLANGE_FEATURE_SUCCESS) {
        printf("Failed to postprocess!");
        return ;
    }

    ret = mlangeYoloV8Feature.resultToImg(img, res);
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
    while ((opt = getopt(argc, argv, "hm:b:c:i:o:g:n")) != -1) {
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
    
    test_yolo8_qnn(model_file_path, qnn_backend_path, qnn_graph_name, input_coco_file_paths_arg, input_file_paths_arg, output_file_paths_arg, YOLO_DETECT_V8);

    printf("Test Yolov8\n");

    return 0;
}