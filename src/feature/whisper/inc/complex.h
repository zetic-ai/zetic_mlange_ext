#pragma once

#include <cmath>

class Complex {
public:
    float real;
    float imag;

    explicit Complex(float r = 0.0f, float i = 0.0f) : real(r), imag(i) {}

    Complex operator+(const Complex& other) const {
        return Complex(real + other.real, imag + other.imag);
    }

    Complex operator-(const Complex& other) const {
        return Complex(real - other.real, imag - other.imag);
    }

    Complex operator*(const Complex& other) const {
        return Complex(
                real * other.real - imag * other.imag,
                real * other.imag + imag * other.real
        );
    }

    float abs() const {
        return std::sqrt(real * real + imag * imag);
    }
};