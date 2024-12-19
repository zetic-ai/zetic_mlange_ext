#pragma once


#include <vector>
#include "landmark.h"

namespace ZeticMLange {
    class FaceLandmarkResult {
    public:
        std::vector<Landmark> landmarks;
        float confidence;
    };
}