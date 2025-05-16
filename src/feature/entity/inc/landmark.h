#pragma once

namespace ZeticMLange {
    // Landmark class represent coordinate data of facial landmark in ROI to range of [0, 1].
    class Landmark {
    public:
        Landmark(float x, float y, float z): x{x}, y{y}, z{z} {}

        float x;
        float y;
        float z;
    };
}