#pragma once

#include "box.h"

namespace ZeticMLange {
    class FaceDetectionResult {
    public:
        Box bbox;
        float score;
    };
}
