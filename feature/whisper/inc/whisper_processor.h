#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include "complex.h"

class WhisperProcessor {
public:
    WhisperProcessor(int n_fft = 400, int hop_length = 160);

    std::vector<float> process(const std::vector<float>& audio_input);

private:
    const int num_frequency_bins = 1 + 400 / 2;
    const int num_mel_filters = 80;
    const float min_frequency = 0.0f;
    const float max_frequency = 8000.0f;
    const int sampling_rate = 16000;
    const std::string norm = "slaney";
    const std::string mel_scale = "slaney";
    const int n_fft;
    const int hop_length;
    const std::vector<std::vector<float>> mel_filters;
    std::vector<float> hann_window;
    std::vector<std::vector<float>> dft_coefficients_real;
    std::vector<std::vector<float>> dft_coefficients_imag;

    static std::vector<std::vector<float>> melFilterBank(
            int num_frequency_bins,
            int num_mel_filters,
            float min_frequency,
            float max_frequency,
            int sampling_rate,
            const std::string& norm = "",
            const std::string& mel_scale = "slaney",
            bool triangularize_in_mel_space = false
    );

    static float hertzToMel(float hz, const std::string& mel_scale);
    static float melToHertz(float mel, const std::string& mel_scale);
    static std::vector<std::vector<float>> createTriangularFilterBank(const std::vector<float>& fft_freqs, const std::vector<float>& filter_freqs, int num_mel_filters);
    std::vector<float> pad(const std::vector<float>& speech, int max_length = 480000, float padding_value = 0.0f);
    std::vector<float> reflectPad(const std::vector<float>& input);
    std::vector<std::vector<Complex>> computeSTFT(const std::vector<float>& waveform);
    std::vector<std::vector<float>> computeMagnitudes(const std::vector<std::vector<Complex>>& stft);
    std::vector<float> applyMelFilterbank(const std::vector<std::vector<float>>& magnitudes);
    std::vector<float> normalizeLogMel(const std::vector<float>& melSpec);
};
