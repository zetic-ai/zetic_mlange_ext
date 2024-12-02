#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include "complex.h"

class WhisperProcessor {
public:
    WhisperProcessor(
            int nFft = 400,
            int hopLength = 160
    ) : nFft(nFft),
        hopLength(hopLength),
        melFilters(melFilterBank(
                1 + 400 / 2,  // numFrequencyBins
                80,           // numMelFilters
                0.0f,        // minFrequency
                8000.0f,     // maxFrequency
                16000,       // samplingRate
                "slaney",    // norm
                "slaney"     // melScale
        )),
        hannWindow(nFft),
        dftCoefficientsReal(nFft, std::vector<float>(nFft)),
        dftCoefficientsImag(nFft, std::vector<float>(nFft))
    {
        // Initialize Hann window
        for (int i = 0; i < nFft; i++) {
            hannWindow[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / nFft));
        }

        // Initialize DFT coefficients
        for (int k = 0; k < nFft; k++) {
            for (int t = 0; t < nFft; t++) {
                double angle = -2.0 * M_PI * k * t / nFft;
                dftCoefficientsReal[k][t] = static_cast<float>(std::cos(angle));
                dftCoefficientsImag[k][t] = static_cast<float>(std::sin(angle));
            }
        }
    }

    std::vector<float> process(const std::vector<float>& audioInput) {
        // Pad and reflect the input
        auto paddedInput = reflectPad(pad(audioInput));

        // Compute Short-Time Fourier Transform
        auto stft = computeSTFT(paddedInput);

        // Compute magnitude spectrograms
        auto magnitudes = computeMagnitudes(stft);

        // Apply mel filterbank
        auto melSpec = applyMelFilterbank(magnitudes);

        // Normalize and take log
        return normalizeLogMel(melSpec);
    }

private:
    const int nFft;
    const int hopLength;
    const std::vector<std::vector<float>> melFilters;
    std::vector<float> hannWindow;
    std::vector<std::vector<float>> dftCoefficientsReal;
    std::vector<std::vector<float>> dftCoefficientsImag;

    static std::vector<std::vector<float>> melFilterBank(
            int numFrequencyBins,
            int numMelFilters,
            float minFrequency,
            float maxFrequency,
            int samplingRate,
            const std::string& norm = "",
            const std::string& melScale = "slaney",
            bool triangularizeInMelSpace = false
    );

    static float hertzToMel(float hz, const std::string& melScale);
    static float melToHertz(float mel, const std::string& melScale);
    static std::vector<std::vector<float>> createTriangularFilterBank(const std::vector<float>& fftFreqs, const std::vector<float>& filterFreqs, int numMelFilters);
    std::vector<float> pad(const std::vector<float>& speech, int maxLength = 480000, float paddingValue = 0.0f);
    std::vector<float> reflectPad(const std::vector<float>& input);
    std::vector<std::vector<Complex>> computeSTFT(const std::vector<float>& waveform);
    std::vector<std::vector<float>> computeMagnitudes(const std::vector<std::vector<Complex>>& stft);
    std::vector<float> applyMelFilterbank(const std::vector<std::vector<float>>& magnitudes);
    std::vector<float> normalizeLogMel(const std::vector<float>& melSpec);

};