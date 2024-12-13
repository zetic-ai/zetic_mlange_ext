#pragma once

#include "../../entity/inc/box.h"

namespace ZeticMLange {
    class FaceDetectionResult {
    public:
        Box bounding_box;
        float score;
    };
}
