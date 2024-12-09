#pragma once

#include "feature_opencv.h"

namespace ZeticMLange {
    class Box {
    public:
        Box() : Box(0, 0, 0, 0) {}
        Box(float x_min, float y_min, float x_max, float y_max) : x_min{x_min}, y_min{y_min}, x_max{x_max}, y_max{y_max} {}
        Box(const cv::Rect2f rect) : x_min{rect.x - (rect.width / 2)}, y_min{rect.y - (rect.height / 2)}, x_max{rect.x + (rect.width / 2)}, y_max{rect.y + (rect.height / 2)} {}
        Box(const Box& other) : x_min{other.x_min}, y_min {other.y_min}, x_max {other.x_max}, y_max{other.y_max} {}

        Box operator*(float factor) const {
            return Box(*this) *= factor;
        }
        Box& operator*=(float factor) {
            x_min *= factor;
            y_min *= factor;
            x_max *= factor;
            y_max *= factor;
            return *this;
        }

        Box operator/(float factor) {
            return Box(*this) /= factor;
        }
        Box& operator/=(float factor) {
            x_min /= factor;
            y_min /= factor;
            x_max /= factor;
            y_max /= factor;
            return *this;
        }

        bool isValid() const {
            return (x_min < x_max) && (y_min < y_max);
        }

        Box intersect(const Box& other) const {
            if (isValid()) {
                return Box(
                        std::max(x_min, other.x_min),
                        std::max(y_min, other.y_min),
                        std::min(x_max, other.x_max),
                        std::min(y_max, other.y_max));
            } else {
                return Box(0,0,0,0);
            }
        }

        float area() const {
            return (x_max - x_min) * (y_max - y_min);
        }

        float overlapSimilarity(const Box& other) const {
            Box intersection = intersect(other);
            if (!intersection.isValid())
                return 0.0f;
            float intersection_area = intersection.area();
            float denominator = (area()) + (other.area()) - intersection_area;
            if (denominator > 0.0f) {
                return intersection_area / denominator;
            } else {
                return 0.0f;
            }
        }

        float x_min;
        float y_min;
        float x_max;
        float y_max;
    };
}
