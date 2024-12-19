#pragma once

#include "box.h"

namespace ZeticMLange {
    class FaceDetectionResult {
    public:
        Box bounding_box;
        float score;
    };
}
