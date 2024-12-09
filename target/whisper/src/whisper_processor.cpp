#include "whisper_processor.h"
#include "dbg_util.h"

WhisperProcessor::WhisperProcessor(int n_fft, int hop_length) : n_fft(n_fft), hop_length(hop_length),
    mel_filters(melFilterBank(
            1 + 400 / 2,  // num_frequency_bins
            80,           // num_mel_filters
            0.0f,        // min_frequency
            8000.0f,     // max_frequency
            16000,       // sampling_rate
            "slaney",    // norm
            "slaney"     // mel_scale
    )),
    hann_window(n_fft),
    dft_coefficients_real(n_fft, std::vector<float>(n_fft)),
    dft_coefficients_imag(n_fft, std::vector<float>(n_fft))
{
    // Initialize Hann window
    for (int i = 0; i < n_fft; i++) {
        hann_window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / n_fft));
    }
    // Initialize DFT coefficients
    for (int k = 0; k < n_fft; k++) {
        for (int t = 0; t < n_fft; t++) {
            double angle = -2.0 * M_PI * k * t / n_fft;
            dft_coefficients_real[k][t] = static_cast<float>(std::cos(angle));
            dft_coefficients_imag[k][t] = static_cast<float>(std::sin(angle));
        }
    }
}

std::vector<float> WhisperProcessor::process(const std::vector<float>& audio_input) {
    // Pad and reflect the input
    auto padded_input = reflectPad(pad(audio_input));
    // Compute Short-Time Fourier Transform
    auto stft = computeSTFT(padded_input);
    // Compute magnitude spectrograms
    auto magnitudes = computeMagnitudes(stft);
    // Apply mel filterbank
    auto mel_spec = applyMelFilterbank(magnitudes);
    // Normalize and take log
    return normalizeLogMel(mel_spec);
}

std::vector<std::vector<float>> WhisperProcessor::melFilterBank(int num_frequency_bins, int num_mel_filters, float min_frequency,
                                float max_frequency, int sampling_rate, const std::string &norm,
                                const std::string &mel_scale, bool triangularize_in_mel_space) {
    if (!norm.empty() && norm != "slaney") {
        ERRLOG("norm must be empty or \"slaney\"");
        return std::vector<std::vector<float>>();
    }

    // Center points of the triangular mel filters
    float mel_min = hertzToMel(min_frequency, mel_scale);
    float mel_max = hertzToMel(max_frequency, mel_scale);

    std::vector<float> mel_freqs(num_mel_filters + 2);
    for (int i = 0; i < num_mel_filters + 2; i++) {
        mel_freqs[i] = mel_min + (mel_max - mel_min) * i / (num_mel_filters + 1);
    }

    std::vector<float> filter_freqs(num_mel_filters + 2);
    for (int i = 0; i < mel_freqs.size(); i++) {
        filter_freqs[i] = melToHertz(mel_freqs[i], mel_scale);
    }

    std::vector<float> fft_freqs(num_frequency_bins);
    if (triangularize_in_mel_space) {
        float fft_bin_width = static_cast<float>(sampling_rate) / (num_frequency_bins * 2);
        for (int i = 0; i < num_frequency_bins; i++) {
            fft_freqs[i] = hertzToMel(fft_bin_width * i, mel_scale);
        }
    } else {
        for (int i = 0; i < num_frequency_bins; i++) {
            fft_freqs[i] = (static_cast<float>(sampling_rate) / 2) * i / (num_frequency_bins - 1);
        }
    }

    auto mel_filters = createTriangularFilterBank(fft_freqs, filter_freqs, num_mel_filters);

    for (int i = 0; i < num_mel_filters; i++) {
        float enorm = 2.0f / (filter_freqs[i + 2] - filter_freqs[i]);
        for (float &value: mel_filters[i]) {
            value *= enorm;
        }
    }

    // Check for zero filters
    bool has_zero_filter = false;
    for (const auto &filter: mel_filters) {
        bool all_zero = true;
        for (float value: filter) {
            if (value != 0.0f) {
                all_zero = false;
                break;
            }
        }
        if (all_zero) {
            has_zero_filter = true;
            break;
        }
    }

    if (has_zero_filter) {
        DBGLOG("Warning: At least one mel filter has all zero values. The value for num_mel_filters (%d) may be set too high. Or, the value for num_frequency_bins (%d) may be set too low.", num_mel_filters, num_frequency_bins);
    }

    return mel_filters;
}

float WhisperProcessor::hertzToMel(float hz, const std::string &mel_scale) {
    if (mel_scale == "slaney") {
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

float WhisperProcessor::melToHertz(float mel, const std::string &mel_scale) {
    if (mel_scale == "slaney") {
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

std::vector<std::vector<float>> WhisperProcessor::createTriangularFilterBank(const std::vector<float> &fft_freqs,
                                             const std::vector<float> &filter_freqs,
                                             int num_mel_filters) {
    std::vector<std::vector<float>> mel_filters(
            num_mel_filters,
            std::vector<float>(fft_freqs.size(), 0.0f)
    );

    for (int i = 0; i < num_mel_filters; i++) {
        float left = filter_freqs[i];
        float center = filter_freqs[i + 1];
        float right = filter_freqs[i + 2];

        for (size_t j = 0; j < fft_freqs.size(); j++) {
            float freq = fft_freqs[j];
            if (freq >= left && freq <= right) {
                if (freq <= center) {
                    mel_filters[i][j] = (freq - left) / (center - left);
                } else {
                    mel_filters[i][j] = (right - freq) / (right - center);
                }
            }
        }
    }

    return mel_filters;
}

std::vector<float>
WhisperProcessor::pad(const std::vector<float> &speech, int maxLength, float paddingValue) {
    size_t currentLength = speech.size();

    if (currentLength == maxLength) {
        return speech;
    }

    if (currentLength > maxLength) {
        return {speech.begin(), speech.begin() + maxLength};
    }

    std::vector<float> padded_array(maxLength, paddingValue);
    std::copy(speech.begin(), speech.end(), padded_array.begin());
    return padded_array;
}

std::vector<float> WhisperProcessor::reflectPad(const std::vector<float> &input) {
    int pad = n_fft / 2;
    size_t input_length = input.size();
    size_t padded_length = input_length + 2 * pad;
    std::vector<float> padded_array(padded_length);

    std::copy(input.begin(), input.end(), padded_array.begin() + pad);

    for (int i = 0; i < pad; i++) {
        padded_array[pad - 1 - i] = input[i + 1];
    }

    for (int i = 0; i < pad; i++) {
        padded_array[pad + input_length + i] = input[input_length - i - 1];
    }

    return padded_array;
}

std::vector<std::vector<Complex>> WhisperProcessor::computeSTFT(const std::vector<float> &waveform) {
    size_t num_frames = (waveform.size() - n_fft) / hop_length + 1;
    int frequency_bins = n_fft / 2 + 1;

    std::vector<std::vector<Complex>> result(
            frequency_bins,
            std::vector<Complex>(num_frames)
    );

    std::vector<float> windowed_frame(n_fft);

    for (int frame = 0; frame < num_frames; frame++) {
        int start = frame * hop_length;

        // Apply window
        for (int i = 0; i < n_fft; i++) {
            windowed_frame[i] = waveform[start + i] * hann_window[i];
        }

        // Compute DFT for each frequency bin
        for (int k = 0; k < frequency_bins; k++) {
            float sum_real = 0.0f;
            float sum_imag = 0.0f;

            const auto &cos_row = dft_coefficients_real[k];
            const auto &sin_row = dft_coefficients_imag[k];

            // Manual loop unrolling
            int t = 0;
            while (t < n_fft - 3) {
                float sample0 = windowed_frame[t];
                float sample1 = windowed_frame[t + 1];
                float sample2 = windowed_frame[t + 2];
                float sample3 = windowed_frame[t + 3];

                sum_real += sample0 * cos_row[t] +
                           sample1 * cos_row[t + 1] +
                           sample2 * cos_row[t + 2] +
                           sample3 * cos_row[t + 3];
                sum_imag += sample0 * sin_row[t] +
                           sample1 * sin_row[t + 1] +
                           sample2 * sin_row[t + 2] +
                           sample3 * sin_row[t + 3];

                t += 4;
            }

            // Handle remaining elements
            while (t < n_fft) {
                float sample = windowed_frame[t];
                sum_real += sample * cos_row[t];
                sum_imag += sample * sin_row[t];
                t++;
            }

            result[k][frame] = Complex(sum_real, sum_imag);
        }
    }

    return result;
}

std::vector<std::vector<float>> WhisperProcessor::computeMagnitudes(const std::vector<std::vector<Complex>> &stft) {
    std::vector<std::vector<float>> magnitudes;
    for (const auto &frame: stft) {
        std::vector<float> frame_magnitudes(frame.size() - 1);
        for (size_t i = 0; i < frame.size() - 1; i++) {
            float magnitude = frame[i].abs();
            frame_magnitudes[i] = magnitude * magnitude;
        }
        magnitudes.push_back(frame_magnitudes);
    }
    return magnitudes;
}

std::vector<float> WhisperProcessor::applyMelFilterbank(const std::vector<std::vector<float>> &magnitudes) {
    {
        size_t num_frames = magnitudes[0].size();
        size_t numMels = mel_filters.size();
        size_t num_freq_bins = mel_filters[0].size();

        std::vector<float> result(num_frames * numMels);

        for (int mel = 0; mel < numMels; mel++) {
            for (int frame = 0; frame < num_frames; frame++) {
                float sum = 0.0f;
                for (int freq = 0; freq < num_freq_bins; freq++) {
                    sum += mel_filters[mel][freq] * magnitudes[freq][frame];
                }
                result[mel * num_frames + frame] = sum;
            }
        }

        return result;
    }
}

std::vector<float> WhisperProcessor::normalizeLogMel(const std::vector<float> &mel_spec) {
    std::vector<float> log_spec(mel_spec.size());

    // Calculate log10 values with clamp
    for (size_t i = 0; i < mel_spec.size(); i++) {
        log_spec[i] = std::log10(std::max(mel_spec[i], 1e-10f));
    }

    // Find maximum value
    float max_val = *std::max_element(log_spec.begin(), log_spec.end());

    // Apply maximum value threshold and normalization
    for (float &value: log_spec) {
        value = std::max(value, max_val - 8.0f);
        value = (value + 4.0f) / 4.0f;
    }

    return log_spec;
}
