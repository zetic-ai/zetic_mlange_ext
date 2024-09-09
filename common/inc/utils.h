#pragma once

#include <cmath>

namespace ZeticMLange {
    template<class T>
    T sigmoid(T x) {
        return 1.f / (1.f + std::exp(-x));
    }
}