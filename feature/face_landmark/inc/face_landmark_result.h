#pragma once


#include <vector>
#include "../../entity/inc/landmark.h"

namespace ZeticMLange {
    class FaceLandmarkResult {
    public:
        std::vector<Landmark> landmarks;
        float confidence;
    };
}