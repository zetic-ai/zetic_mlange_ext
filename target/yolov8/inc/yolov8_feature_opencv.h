#pragma once

#include <vector>

#include "feature_opencv.h"
#include "zetic_feature_types.h"

#define MAX_YOLO_RESULT_COUNT 20

enum YOLO_MODEL_TYPE {
    YOLO_DETECT_V8 = 1,
    YOLO_POSE = 2,
    YOLO_CLS = 3,
};

typedef struct _DL_RESULT {
    int classId;
    float confidence;
    cv::Rect box;
    std::vector<cv::Point2f> keyPoints;
} DL_RESULT;

typedef struct {
    int32_t classId;
    float confidence;
    int32_t x;
    int32_t y;
    int32_t width;
    int32_t height;
} DLResultC;

typedef struct _DL_PARAM {
    std::vector<int> imgSize = { 640, 640 };
    float rectConfidenceThreshold = 0.5;
    float iouThreshold = 0.5;
    int	keyPointsNum = 2;                       //Note:kpt number for pose
} DL_PARAM;

class ZeticMLangeYoloV8Feature {
public:
    ZeticMLangeYoloV8Feature(YOLO_MODEL_TYPE yolo_model_type, const char* coco_file_path);
    ~ZeticMLangeYoloV8Feature();

    Zetic_MLange_Feature_Result_t getByteArrayFromImage(cv::Mat& input_img, int8_t* blob);
    Zetic_MLange_Feature_Result_t preprocess(cv::Mat& input_img, cv::Mat& output_image);
    Zetic_MLange_Feature_Result_t postprocess(std::vector<DL_RESULT>& output_dl_result, void* output);

    Zetic_MLange_Feature_Result_t resultToImg(cv::Mat& img, std::vector<DL_RESULT> res);
    
private:
    Zetic_MLange_Feature_Result_t readCocoYaml(const char* coco_file_path);

    Zetic_MLange_Feature_Result_t detectorResultToImg(cv::Mat& img, std::vector<DL_RESULT> res);
    Zetic_MLange_Feature_Result_t classifierResultToImg(cv::Mat& img, std::vector<DL_RESULT> res);
    

    std::vector<std::string> classes{};
    int yolo_model_type;
    MLangeFeatureOpenCV* mlange_feature_opencv;
    DL_PARAM dl_params;
    float xResizeScale;     // For letterbox scale
    float yResizeScale;     // For letterbox scale
};
