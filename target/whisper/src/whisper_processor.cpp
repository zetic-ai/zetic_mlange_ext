#include "whisper_processor.h"

std::vector<std::vector<float>>
WhisperProcessor::melFilterBank(int numFrequencyBins, int numMelFilters, float minFrequency,
                                float maxFrequency, int samplingRate, const std::string &norm,
                                const std::string &melScale, bool triangularizeInMelSpace) {
    if (!norm.empty() && norm != "slaney") {
        throw std::runtime_error("norm must be empty or \"slaney\"");
    }

    // Center points of the triangular mel filters
    float melMin = hertzToMel(minFrequency, melScale);
    float melMax = hertzToMel(maxFrequency, melScale);

    std::vector<float> melFreqs(numMelFilters + 2);
    for (int i = 0; i < numMelFilters + 2; i++) {
        melFreqs[i] = melMin + (melMax - melMin) * i / (numMelFilters + 1);
    }

    std::vector<float> filterFreqs(numMelFilters + 2);
    for (int i = 0; i < melFreqs.size(); i++) {
        filterFreqs[i] = melToHertz(melFreqs[i], melScale);
    }

    std::vector<float> fftFreqs(numFrequencyBins);
    if (triangularizeInMelSpace) {
        float fftBinWidth = static_cast<float>(samplingRate) / (numFrequencyBins * 2);
        for (int i = 0; i < numFrequencyBins; i++) {
            fftFreqs[i] = hertzToMel(fftBinWidth * i, melScale);
        }
    } else {
        for (int i = 0; i < numFrequencyBins; i++) {
            fftFreqs[i] = (static_cast<float>(samplingRate) / 2) * i / (numFrequencyBins - 1);
        }
    }

    auto melFilters = createTriangularFilterBank(fftFreqs, filterFreqs, numMelFilters);

    for (int i = 0; i < numMelFilters; i++) {
        float enorm = 2.0f / (filterFreqs[i + 2] - filterFreqs[i]);
        for (float &value: melFilters[i]) {
            value *= enorm;
        }
    }

    // Check for zero filters
    bool hasZeroFilter = false;
    for (const auto &filter: melFilters) {
        bool allZero = true;
        for (float value: filter) {
            if (value != 0.0f) {
                allZero = false;
                break;
            }
        }
        if (allZero) {
            hasZeroFilter = true;
            break;
        }
    }

    if (hasZeroFilter) {
        std::cout << "Warning: At least one mel filter has all zero values. "
                  << "The value for numMelFilters (" << numMelFilters
                  << ") may be set too high. "
                  << "Or, the value for numFrequencyBins (" << numFrequencyBins
                  << ") may be set too low."
                  << std::endl;
    }

    return melFilters;
}

float WhisperProcessor::hertzToMel(float hz, const std::string &melScale) {
    if (melScale == "slaney") {
        const float f_min = 0.0f;
        const float f_sp = 200.0f / 3.0f;
        const float min_log_hz = 1000.0f;
        const float min_log_mel = (min_log_hz - f_min) / f_sp;
        const float logstep = std::log(6.4f) / 27.0f;

        if (hz >= min_log_hz) {
            return min_log_mel + (std::log(hz / min_log_hz) / logstep);
        } else {
            return (hz - f_min) / f_sp;
        }
    } else {  // "htk"
        return 2595.0f * std::log10(1.0f + hz / 700.0f);
    }
}

float WhisperProcessor::melToHertz(float mel, const std::string &melScale) {
    if (melScale == "slaney") {
        const float f_min = 0.0f;
        const float f_sp = 200.0f / 3.0f;
        const float min_log_hz = 1000.0f;
        const float min_log_mel = (min_log_hz - f_min) / f_sp;
        const float logstep = std::log(6.4f) / 27.0f;

        if (mel >= min_log_mel) {
            return min_log_hz * std::exp(logstep * (mel - min_log_mel));
        } else {
            return f_min + f_sp * mel;
        }
    } else {  // "htk"
        return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
    }
}

std::vector<std::vector<float>>
WhisperProcessor::createTriangularFilterBank(const std::vector<float> &fftFreqs,
                                             const std::vector<float> &filterFreqs,
                                             int numMelFilters) {
    std::vector<std::vector<float>> melFilters(
            numMelFilters,
            std::vector<float>(fftFreqs.size(), 0.0f)
    );

    for (int i = 0; i < numMelFilters; i++) {
        float left = filterFreqs[i];
        float center = filterFreqs[i + 1];
        float right = filterFreqs[i + 2];

        for (size_t j = 0; j < fftFreqs.size(); j++) {
            float freq = fftFreqs[j];
            if (freq >= left && freq <= right) {
                if (freq <= center) {
                    melFilters[i][j] = (freq - left) / (center - left);
                } else {
                    melFilters[i][j] = (right - freq) / (right - center);
                }
            }
        }
    }

    return melFilters;
}

std::vector<float>
WhisperProcessor::pad(const std::vector<float> &speech, int maxLength, float paddingValue) {
    int currentLength = speech.size();

    if (currentLength == maxLength) {
        return speech;
    }

    if (currentLength > maxLength) {
        return {speech.begin(), speech.begin() + maxLength};
    }

    std::vector<float> paddedArray(maxLength, paddingValue);
    std::copy(speech.begin(), speech.end(), paddedArray.begin());
    return paddedArray;
}

std::vector<float> WhisperProcessor::reflectPad(const std::vector<float> &input) {
    int pad = nFft / 2;
    int inputLength = input.size();
    int paddedLength = inputLength + 2 * pad;
    std::vector<float> paddedArray(paddedLength);

    std::copy(input.begin(), input.end(), paddedArray.begin() + pad);

    for (int i = 0; i < pad; i++) {
        paddedArray[pad - 1 - i] = input[i + 1];
    }

    for (int i = 0; i < pad; i++) {
        paddedArray[pad + inputLength + i] = input[inputLength - i - 1];
    }

    return paddedArray;
}

std::vector<std::vector<Complex>>
WhisperProcessor::computeSTFT(const std::vector<float> &waveform) {
    int numFrames = (waveform.size() - nFft) / hopLength + 1;
    int frequencyBins = nFft / 2 + 1;

    std::vector<std::vector<Complex>> result(
            frequencyBins,
            std::vector<Complex>(numFrames)
    );

    std::vector<float> windowedFrame(nFft);

    for (int frame = 0; frame < numFrames; frame++) {
        int start = frame * hopLength;

        // Apply window
        for (int i = 0; i < nFft; i++) {
            windowedFrame[i] = waveform[start + i] * hannWindow[i];
        }

        // Compute DFT for each frequency bin
        for (int k = 0; k < frequencyBins; k++) {
            float sumReal = 0.0f;
            float sumImag = 0.0f;

            const auto &cosRow = dftCoefficientsReal[k];
            const auto &sinRow = dftCoefficientsImag[k];

            // Manual loop unrolling
            int t = 0;
            while (t < nFft - 3) {
                float sample0 = windowedFrame[t];
                float sample1 = windowedFrame[t + 1];
                float sample2 = windowedFrame[t + 2];
                float sample3 = windowedFrame[t + 3];

                sumReal += sample0 * cosRow[t] +
                           sample1 * cosRow[t + 1] +
                           sample2 * cosRow[t + 2] +
                           sample3 * cosRow[t + 3];
                sumImag += sample0 * sinRow[t] +
                           sample1 * sinRow[t + 1] +
                           sample2 * sinRow[t + 2] +
                           sample3 * sinRow[t + 3];

                t += 4;
            }

            // Handle remaining elements
            while (t < nFft) {
                float sample = windowedFrame[t];
                sumReal += sample * cosRow[t];
                sumImag += sample * sinRow[t];
                t++;
            }

            result[k][frame] = Complex(sumReal, sumImag);
        }
    }

    return result;
}

std::vector<std::vector<float>>
WhisperProcessor::computeMagnitudes(const std::vector<std::vector<Complex>> &stft) {
    std::vector<std::vector<float>> magnitudes;
    for (const auto &frame: stft) {
        std::vector<float> frameMagnitudes(frame.size() - 1);
        for (size_t i = 0; i < frame.size() - 1; i++) {
            float magnitude = frame[i].abs();
            frameMagnitudes[i] = magnitude * magnitude;
        }
        magnitudes.push_back(frameMagnitudes);
    }
    return magnitudes;
}

std::vector<float>
WhisperProcessor::applyMelFilterbank(const std::vector<std::vector<float>> &magnitudes) {
    {
        int numFrames = magnitudes[0].size();
        int numMels = melFilters.size();
        int numFreqBins = melFilters[0].size();

        std::vector<float> result(numFrames * numMels);

        for (int mel = 0; mel < numMels; mel++) {
            for (int frame = 0; frame < numFrames; frame++) {
                float sum = 0.0f;
                for (int freq = 0; freq < numFreqBins; freq++) {
                    sum += melFilters[mel][freq] * magnitudes[freq][frame];
                }
                result[mel * numFrames + frame] = sum;
            }
        }

        return result;
    }
}

std::vector<float> WhisperProcessor::normalizeLogMel(const std::vector<float> &melSpec) {
    std::vector<float> logSpec(melSpec.size());

    // Calculate log10 values with clamp
    for (size_t i = 0; i < melSpec.size(); i++) {
        logSpec[i] = std::log10(std::max(melSpec[i], 1e-10f));
    }

    // Find maximum value
    float maxVal = *std::max_element(logSpec.begin(), logSpec.end());

    // Apply maximum value threshold and normalization
    for (float &value: logSpec) {
        value = std::max(value, maxVal - 8.0f);
        value = (value + 4.0f) / 4.0f;
    }

    return logSpec;
}
