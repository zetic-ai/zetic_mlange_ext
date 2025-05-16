#pragma once

#include "../../../vision/opencv/inc/feature_opencv.h"

namespace ZeticMLange {
    class Box {
    public:
        Box();
        Box(float x_min, float y_min, float x_max, float y_max);
        Box(const cv::Rect2f rect);
        Box(const Box& other);

        Box operator*(float factor) const;
        Box& operator*=(float factor);
        Box operator/(float factor);
        Box& operator/=(float factor);

        bool isValid() const;
        Box intersect(const Box& other) const;
        float area() const;
        float overlapSimilarity(const Box& other) const;

        float x_min;
        float y_min;
        float x_max;
        float y_max;
    };
}
