#include "vad-segmenter-generator.h"
#include "whisper.h"
#include <stdexcept>
#include <algorithm>
#include <vector>

struct VadSegmenterGenerator::Impl {
    enum class State { SEARCHING, IN_SPEECH };

    Params params;
    struct whisper_vad_context * vctx = nullptr;
    State state = State::SEARCHING;
    int sample_rate = WHISPER_SAMPLE_RATE;
    int n_window = 512;
    int padding_samples;
    int min_silence_samples;
    int history_samples;
    std::vector<float> pcm_history;
    std::vector<float> pcm_buffer;
    size_t history_idx = 0;
    int samples_since_vad = 0;
    int speech_frames = 0;
    int silence_frames = 0;

    Impl(const Params& params) : params(params) {
        struct whisper_vad_context_params ctx_params = whisper_vad_default_context_params();
        vctx = whisper_vad_init_from_file_with_params(params.vad_model_path.c_str(), ctx_params);
        if (!vctx) throw std::runtime_error("VAD init failed");

        padding_samples = (params.padding_ms * sample_rate) / 1000;
        min_silence_samples = (params.min_silence_ms * sample_rate) / 1000;
        
        // We will keep a history of the last 2 seconds for VAD context
        history_samples = 2.0f * sample_rate;
        pcm_history.resize(history_samples, 0.0f);
    }

    ~Impl() {
        if (vctx) whisper_vad_free(vctx);
    }

    int speech_samples_needed() {
        return (params.min_speech_ms * sample_rate) / 1000;
    }

    bool run_vad() {
        float window[1536];
        for (int i = 0; i < 1536; ++i) {
            int idx = (history_idx - 1536 + i + history_samples) % history_samples;
            window[i] = pcm_history[idx];
        }
        if (!whisper_vad_detect_speech(vctx, window, 1536)) return false;

        float* probs = whisper_vad_probs(vctx);
        int n = whisper_vad_n_probs(vctx);
        return probs[n-1] > params.threshold;
    }

    std::optional<std::vector<float>> feed(const std::vector<float>& pcm_chunk) {
        for (float s : pcm_chunk) {
            pcm_history[history_idx] = s;
            history_idx = (history_idx + 1) % history_samples;
            
            if (state == State::IN_SPEECH) {
                pcm_buffer.push_back(s);
            }

            samples_since_vad++;
            if (samples_since_vad >= n_window) {
                samples_since_vad = 0;
                bool is_speech = run_vad();
                
                if (state == State::SEARCHING) {
                    if (is_speech) {
                        speech_frames++;
                        if (speech_samples_needed() <= speech_frames * n_window) {
                            state = State::IN_SPEECH;
                            pcm_buffer.clear();
                            int n_pre = padding_samples;
                            for (int i = 0; i < n_pre; ++i) {
                                int idx = (history_idx - n_pre + i + history_samples) % history_samples;
                                pcm_buffer.push_back(pcm_history[idx]);
                            }
                            silence_frames = 0;
                        }
                    } else {
                        speech_frames = 0;
                    }
                } else {
                    if (!is_speech) {
                        silence_frames++;
                        if (silence_frames * n_window >= min_silence_samples) {
                            state = State::SEARCHING;
                            speech_frames = 0;
                            auto result = pcm_buffer;
                            pcm_buffer.clear();
                            return result;
                        }
                    } else {
                        silence_frames = 0;
                    }
                }
            }
        }
        return std::nullopt;
    }

    std::optional<std::vector<float>> flush() {
        if (state == State::IN_SPEECH && !pcm_buffer.empty()) {
            auto result = std::move(pcm_buffer);
            state = State::SEARCHING;
            return result;
        }
        return std::nullopt;
    }
};

VadSegmenterGenerator::VadSegmenterGenerator(const Params& params)
    : pimpl(std::make_unique<Impl>(params)) {}

VadSegmenterGenerator::~VadSegmenterGenerator() = default;

std::optional<std::vector<float>> VadSegmenterGenerator::feed(const std::vector<float>& pcm_chunk) {
    return pimpl->feed(pcm_chunk);
}

std::optional<std::vector<float>> VadSegmenterGenerator::flush() {
    return pimpl->flush();
}
