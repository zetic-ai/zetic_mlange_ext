#include "yolov8_feature.h"
#include "dbg_utils.h"

#include <fstream>
#include <random>

#define YOLO8_OUTPUT_DIM1 8400
#define YOLO8_OUTPUT_DIM2 84

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

Zetic_MLange_Feature_Result_t ZeticMLangeYoloV8Feature::postprocess(std::vector<DL_RESULT>& output_dl_result,
                                                                    void* output)
{
    if (this->yolo_model_type == YOLO_CLS) {
        // Classification branch
        cv::Mat raw_data(1, static_cast<int>(this->classes.size()), CV_32F, output);
        float* data = reinterpret_cast<float*>(raw_data.data);

        for (size_t i = 0; i < this->classes.size(); ++i) {
            DL_RESULT result;
            result.class_id  = static_cast<int>(i);
            result.confidence = data[i];
            output_dl_result.push_back(result);
        }
    }
    else {
        // YOLO detection branch
        // e.g. stride_num = 8400, signal_result_num = 84
        int stride_num        = YOLO8_OUTPUT_DIM1;
        int signal_result_num = YOLO8_OUTPUT_DIM2;

        // Usually YOLOv8 output = 8400 anchors × 84 floats each
        // But if it's "pre-transposed," it is shape [84 x 8400]
        cv::Mat raw_data(signal_result_num, stride_num, CV_32FC1, output);
        // raw_data.rows = signal_result_num (e.g. 84)
        // raw_data.cols = stride_num        (e.g. 8400)

        std::vector<int>   class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        for (int anchor_idx = 0; anchor_idx < stride_num; ++anchor_idx)
        {
            // Rows 0..3: (x, y, w, h)
            float x = raw_data.at<float>(0, anchor_idx);
            float y = raw_data.at<float>(1, anchor_idx);
            float w = raw_data.at<float>(2, anchor_idx);
            float h = raw_data.at<float>(3, anchor_idx);

            // Build a small 1 x #classes Mat out of row [4..(4 + classes.size() - 1)]
            int num_classes = static_cast<int>(this->classes.size());
            cv::Mat scores(1, num_classes, CV_32FC1);

            for (int c = 0; c < num_classes; ++c) {
                scores.at<float>(0, c) = raw_data.at<float>(4 + c, anchor_idx);
            }

            // Find best class score
            double max_class_score = 0.0;
            cv::Point class_id;
            cv::minMaxLoc(scores, nullptr, &max_class_score, nullptr, &class_id);

            if (max_class_score > this->dl_params.rect_confidence_threshold)
            {
                confidences.push_back(static_cast<float>(max_class_score));
                class_ids.push_back(class_id.x);

                // Scale box to your input coordinates
                int left   = static_cast<int>((x - 0.5f * w) * x_resize_scale);
                int top    = static_cast<int>((y - 0.5f * h) * y_resize_scale);
                int width  = static_cast<int>(w * x_resize_scale);
                int height = static_cast<int>(h * y_resize_scale);

                boxes.emplace_back(left, top, width, height);
            }
        }

        // Apply Non-Maximum Suppression
        std::vector<int> nms_result;
        cv::dnn::NMSBoxes(boxes,
                          confidences,
                          this->dl_params.rect_confidence_threshold,
                          this->dl_params.iou_threshold,
                          nms_result);

        // Collect final results
        for (int idx : nms_result) {
            DL_RESULT result;
            result.class_id   = class_ids[idx];
            result.confidence = confidences[idx];
            result.box        = boxes[idx];
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
        // hard-coded YOLO detection color generation value
        cv::Scalar color((re.class_id + 72) * 1717 % 256, (re.class_id + 7) * 33 % 126 + 70, (re.class_id + 47) * 107 % 256);

        cv::rectangle(img, re.box, color, 5);

        float confidence = re.confidence;
        std::cout << std::fixed << std::setprecision(2);
        std::string label = this->classes[re.class_id] + " " +
            std::to_string(confidence).substr(0, std::to_string(confidence).size() - 4);

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

