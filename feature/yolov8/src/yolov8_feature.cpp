#include "yolov8_feature.h"
#include "dbg_utils.h"

#include <fstream>
#include <random>

#define YOLO8_OUTPUT_DIM1 8400
#define YOLO8_OUTPUT_DIM2 28 // iclo test only

#include <algorithm>
std::vector<int> iclo_detected_classes = {2, 7, 8, 14, 16, 17, 18, 19, 21};

// TODO: Need double-check iclo caries classes
//std::vector<int> iclo_caries_classes = {6, 8, 12, 13, 14, 18};

std::vector<int> iclo_caries_classes = {6, 12, 13};

ZeticMLangeYoloV8Feature::ZeticMLangeYoloV8Feature(YOLO_MODEL_TYPE yolo_model_type, const char* coco_file_path) {
    
    Zetic_MLange_Feature_Result_t ret = ZETIC_MLANGE_FEATURE_FAIL;
    ret = this->readCocoYaml(coco_file_path);
    if (ret != ZETIC_MLANGE_FEATURE_SUCCESS) {
        ERRLOG("Failed to read coco yaml file to load Yolov8 model: %s", coco_file_path);
        return ;
    }

    this->yolo_model_type = yolo_model_type;
    if (this->yolo_model_type == YOLO_CLS) {
        DL_PARAM params{ {224, 224} };
        this->dl_params = params;
        
    } else {
        DL_PARAM params;
        params.rect_confidence_threshold = 0.5;
        params.iou_threshold = 0.5;
        params.img_size = { 640, 640 };
        this->dl_params = params;
    }

    this->mlange_feature_opencv = new MLangeFeatureOpenCV();
}

ZeticMLangeYoloV8Feature::~ZeticMLangeYoloV8Feature() {
    delete(this->mlange_feature_opencv);
}

// TODO: We assign delete responsibility to user, possible hazard.
Zetic_MLange_Feature_Result_t ZeticMLangeYoloV8Feature::getByteArrayFromImage(cv::Mat &input_img,
                                                                              int8_t *blob) {
    Zetic_MLange_Feature_Result_t ret = ZETIC_MLANGE_FEATURE_FAIL;
    
    ret = this->mlange_feature_opencv->getByteArrayFromImage(input_img, blob);
    if (ret != ZETIC_MLANGE_FEATURE_SUCCESS) {
        ERRLOG("Failed to get float array from image!");
        return ret;
    }

    return ret;
}

Zetic_MLange_Feature_Result_t ZeticMLangeYoloV8Feature::preprocess(cv::Mat& input_img, cv::Mat& output_image) {
     std::vector<int> input_img_size = this->dl_params.img_size;
    
    if (this->yolo_model_type != YOLO_CLS) {
        x_resize_scale = input_img.cols / (float)input_img_size.at(0);
        y_resize_scale = input_img.rows / (float)input_img_size.at(1);
        return this->mlange_feature_opencv->getLetterBox(input_img, input_img_size, output_image);
    } else {
        return this->mlange_feature_opencv->getCenterCrop(input_img, input_img_size, output_image);
    }
}

Zetic_MLange_Feature_Result_t ZeticMLangeYoloV8Feature::postprocess(std::vector<DL_RESULT>& output_dl_result, void* output) {
    if (this->yolo_model_type == YOLO_CLS) {
        cv::Mat raw_data;
        
        // FP32
        raw_data = cv::Mat(1, (int)this->classes.size(), CV_32F, output);
        float *data = (float *) raw_data.data;

        DL_RESULT result;
        for (int i = 0; i < this->classes.size(); i++) {
            result.class_id = i;
            result.confidence = data[i];
            output_dl_result.push_back(result);
        }
    } else {
        // Hard-coded output dimensions
        int stride_num = YOLO8_OUTPUT_DIM1; //outputNodeDims[1];
        int signal_result_num = YOLO8_OUTPUT_DIM2; //outputNodeDims[2];

        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        
        cv::Mat raw_data = cv::Mat(signal_result_num, stride_num, CV_8SC4, output);

        raw_data = raw_data.t();
        float* data = (float*)(raw_data.data);

        for (int i = 0; i < stride_num; ++i) {
            float* classes_scores = data + 4;
            cv::Mat scores(1, (int)this->classes.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

            bool detected = false;
            // (1) Filter the class
            if (std::find(iclo_detected_classes.begin(), iclo_detected_classes.end(), class_id.x) != iclo_detected_classes.end()) {
                // the detected class is in iclo detected classes
                // TODO: Remove hard-coded score
                if (max_class_score > 0.5f) {
                    detected = true;
                }
            } else if (std::find(iclo_caries_classes.begin(), iclo_caries_classes.end(), class_id.x) != iclo_caries_classes.end()) {
                // the detected class is in iclo caries classes
                // TODO: Remove hard-coded score
                if (max_class_score > 0.3f) {
                    detected = true;
                }
            } else {
                // do nothing
            }

            if (detected) {

                confidences.push_back((float)max_class_score);
                class_ids.push_back(class_id.x);
                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = int((x - 0.5 * w) * x_resize_scale);
                int top = int((y - 0.5 * h) * y_resize_scale);

                int width = int(w * x_resize_scale);
                int height = int(h * y_resize_scale);

                boxes.emplace_back(left, top, width, height);
            }
            data += signal_result_num;
        }

        std::vector<int> nms_result;
        cv::dnn::NMSBoxes(boxes, confidences, this->dl_params.rect_confidence_threshold, this->dl_params.iou_threshold, nms_result);
        for (int i = 0; i < nms_result.size(); ++i) {
            int idx = nms_result[i];
            DL_RESULT result;
            result.class_id = class_ids[idx];
            result.confidence = confidences[idx];
            result.box = boxes[idx];
            output_dl_result.push_back(result);
        }
    }

    return ZETIC_MLANGE_FEATURE_SUCCESS;
}

Zetic_MLange_Feature_Result_t ZeticMLangeYoloV8Feature::resultToImg(cv::Mat& img, std::vector<DL_RESULT> res) {
    if (this->yolo_model_type == YOLO_CLS) { 
        return this->classifierResultToImg(img, res);
    } else {
        return this->detectorResultToImg(img, res);
    }
}

Zetic_MLange_Feature_Result_t ZeticMLangeYoloV8Feature::detectorResultToImg(cv::Mat& img, std::vector<DL_RESULT> res) {

    for (auto& re : res) {

        cv::RNG rng(cv::getTickCount());
        // TODO: Currently Hard-coded RGB value to fix the class color
        cv::Scalar color((re.class_id + 72) * 1717 % 256, (re.class_id + 7) * 33 % 126 + 70, (re.class_id + 47) * 107 % 256);

        cv::rectangle(img, re.box, color, 5);

        float confidence = re.confidence;
        std::cout << std::fixed << std::setprecision(2);

        // iclo label: label only
        std::string label = std::to_string(confidence).substr(0, std::to_string(confidence).size() - 4);

        cv::rectangle(
            img,
            cv::Point(re.box.x, re.box.y - 25),
            cv::Point(re.box.x + (int)label.length() * 15, re.box.y),
            color,
            cv::FILLED
        );

        cv::putText(
            img,
            label,
            cv::Point(re.box.x, re.box.y - 5),
            cv::FONT_HERSHEY_SIMPLEX,
            0.75,
            cv::Scalar(0, 0, 0),
            2
        );
    }

    return ZETIC_MLANGE_FEATURE_SUCCESS;
}


Zetic_MLange_Feature_Result_t ZeticMLangeYoloV8Feature::classifierResultToImg(cv::Mat& img, std::vector<DL_RESULT> res) {
    
    Zetic_MLange_Feature_Result_t ret = ZETIC_MLANGE_FEATURE_FAIL;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 255);

    float positionY = 50;
    for (int i = 0; i < res.size(); i++) {
        int r = dis(gen);
        int g = dis(gen);
        int b = dis(gen);
        cv::putText(img, std::to_string(i) + ":", cv::Point(10, positionY), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(b, g, r), 2);
        cv::putText(img, std::to_string(res.at(i).confidence), cv::Point(70, positionY), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(b, g, r), 2);
        positionY += 50;
    }

    cv::imshow("TEST_CLS", img);


    return ret;
}

Zetic_MLange_Feature_Result_t ZeticMLangeYoloV8Feature::readCocoYaml(const char* coco_file_path) {

    // Open the YAML file
    std::ifstream file(coco_file_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file" << std::endl;
        return ZETIC_MLANGE_FEATURE_FAIL;
    }

    // Read the file line by line
    std::string line;
    std::vector<std::string> lines;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }

    // Find the start and end of the names section
    std::size_t start = 0;
    std::size_t end = 0;
    for (std::size_t i = 0; i < lines.size(); i++) {
        if (lines[i].find("names:") != std::string::npos) {
            start = i + 1;
        }
        else if (start > 0 && lines[i].find(':') == std::string::npos) {
            end = i;
            break;
        }
    }

    // Extract the names
    std::vector<std::string> names;
    for (std::size_t i = start; i < end; i++) {
        std::stringstream ss(lines[i]);
        std::string name;
        std::getline(ss, name, ':');        // Extract the number before the delimiter
        std::getline(ss, name);             // Extract the string after the delimiter
        names.push_back(name);
    }

    this->classes = names;

    return ZETIC_MLANGE_FEATURE_SUCCESS;
}

