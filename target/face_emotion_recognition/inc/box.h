#pragma once

#include "feature_opencv.h"

namespace ZeticMLange {
    class Box {
    public:
        Box() : Box(0, 0, 0, 0) {}
        Box(float xmin, float ymin, float xmax, float ymax) : xmin{xmin}, ymin{ymin}, xmax{xmax}, ymax{ymax} {}
        Box(const cv::Rect2f rect) : xmin{rect.x - (rect.width / 2)}, ymin{rect.y - (rect.height / 2)}, xmax{rect.x + (rect.width / 2)}, ymax{rect.y + (rect.height / 2)} {}
        Box(const Box& other) : xmin{other.xmin}, ymin {other.ymin}, xmax {other.xmax}, ymax{other.ymax} {}

        Box operator*(float factor) const {
            return Box(*this) *= factor;
        }
        Box& operator*=(float factor) {
            xmin *= factor;
            ymin *= factor;
            xmax *= factor;
            ymax *= factor;
            return *this;
        }

        Box operator/(float factor) {
            return Box(*this) /= factor;
        }
        Box& operator/=(float factor) {
            xmin /= factor;
            ymin /= factor;
            xmax /= factor;
            ymax /= factor;
            return *this;
        }

        bool isValid() const {
            return (xmin < xmax) && (ymin < ymax);
        }

        Box intersect(const Box& other) const {
            if (isValid()) {
                return Box(
                        std::max(xmin, other.xmin),
                        std::max(ymin, other.ymin),
                        std::min(xmax, other.xmax),
                        std::min(ymax, other.ymax));
            } else {
                return Box(0,0,0,0);
            }
        }

        float area() const {
            return (xmax - xmin) * (ymax - ymin);
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

        float xmin;
        float ymin;
        float xmax;
        float ymax;
    };
}
