#pragma once

#include "landmark.h"

#include <vector>

namespace ZeticMLange {
    class FaceLandmarkResult {
    public:
        std::vector<Landmark> landmarks;
        float confidence;
    };
}