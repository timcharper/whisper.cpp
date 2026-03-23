#include "vad-segmenter-generator.h"
#include "whisper.h"
#include <stdexcept>
#include <algorithm>
#include <vector>

struct VadSegmenterGenerator::Impl
{
    enum class State
    {
        SEARCHING,
        IN_SPEECH
    };

    Params params;
    struct whisper_vad_context *vctx = nullptr;
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

    Impl(const Params &params) : params(params)
    {
        struct whisper_vad_context_params ctx_params = whisper_vad_default_context_params();
        vctx = whisper_vad_init_from_file_with_params(params.vad_model_path.c_str(), ctx_params);
        if (!vctx)
            throw std::runtime_error("VAD init failed");

        // Use the model's native window size so we step and feed exactly one
        // window per inference call, letting the persistent LSTM state carry
        // temporal context instead of the old 3x overlap warmup hack.
        n_window = whisper_vad_get_n_window(vctx);
        whisper_vad_reset_state(vctx);

        padding_samples = (params.padding_ms * sample_rate) / 1000;
        min_silence_samples = (params.min_silence_ms * sample_rate) / 1000;

        // History must hold at least min_speech_ms + padding_ms so the lookback
        // on speech onset never exceeds the ring buffer size.
        int min_history = speech_samples_needed() + padding_samples + n_window;
        history_samples = std::max(min_history, (int)(2.0f * sample_rate));
        pcm_history.resize(history_samples, 0.0f);
    }

    ~Impl()
    {
        if (vctx)
            whisper_vad_free(vctx);
    }

    int speech_samples_needed()
    {
        return (params.min_speech_ms * sample_rate) / 1000;
    }

    bool run_vad()
    {
        // Feed exactly one native model window (n_window samples) ending at the
        // current history position. The LSTM state persists across calls so the
        // model accumulates context naturally — no overlap warmup needed.
        std::vector<float> window(n_window);
        for (int i = 0; i < n_window; ++i)
        {
            int idx = (history_idx - n_window + i + history_samples) % history_samples;
            window[i] = pcm_history[idx];
        }
        if (!whisper_vad_detect_speech(vctx, window.data(), n_window))
            return false;

        float *probs = whisper_vad_probs(vctx);
        return probs[0] > params.threshold;
    }

    std::optional<std::vector<float>> feed(const std::vector<float> &pcm_chunk)
    {
        for (float s : pcm_chunk)
        {
            pcm_history[history_idx] = s;
            history_idx = (history_idx + 1) % history_samples;

            if (state == State::IN_SPEECH)
            {
                pcm_buffer.push_back(s);
            }

            samples_since_vad++;
            if (samples_since_vad >= n_window)
            {
                samples_since_vad = 0;
                bool is_speech = run_vad();

                if (state == State::SEARCHING)
                {
                    if (is_speech)
                    {
                        speech_frames++;
                        if (speech_samples_needed() <= speech_frames * n_window)
                        {
                            state = State::IN_SPEECH;
                            pcm_buffer.clear();
                            // Go back far enough to include all the speech windows that were
                            // used to confirm onset (which were not buffered while SEARCHING),
                            // plus the pre-speech padding. Without this, the portion of speech
                            // that elapsed during the min_speech_ms confirmation window is lost.
                            int n_pre = speech_frames * n_window + padding_samples;
                            n_pre = std::min(n_pre, history_samples - 1);
                            for (int i = 0; i < n_pre; ++i)
                            {
                                int idx = (history_idx - n_pre + i + history_samples) % history_samples;
                                pcm_buffer.push_back(pcm_history[idx]);
                            }
                            silence_frames = 0;
                        }
                    }
                    else
                    {
                        speech_frames = 0;
                    }
                }
                else
                {
                    if (!is_speech)
                    {
                        silence_frames++;
                        if (silence_frames * n_window >= min_silence_samples)
                        {
                            state = State::SEARCHING;
                            speech_frames = 0;
                            whisper_vad_reset_state(vctx);
                            auto result = pcm_buffer;
                            pcm_buffer.clear();
                            return result;
                        }
                    }
                    else
                    {
                        silence_frames = 0;
                    }
                }
            }
        }
        return std::nullopt;
    }

    std::optional<std::vector<float>> flush()
    {
        if (state == State::IN_SPEECH && !pcm_buffer.empty())
        {
            auto result = std::move(pcm_buffer);
            state = State::SEARCHING;
            whisper_vad_reset_state(vctx);
            return result;
        }
        return std::nullopt;
    }
};

VadSegmenterGenerator::VadSegmenterGenerator(const Params &params)
    : pimpl(std::make_unique<Impl>(params)) {}

VadSegmenterGenerator::~VadSegmenterGenerator() = default;

std::optional<std::vector<float>> VadSegmenterGenerator::feed(const std::vector<float> &pcm_chunk)
{
    return pimpl->feed(pcm_chunk);
}

std::optional<std::vector<float>> VadSegmenterGenerator::flush()
{
    return pimpl->flush();
}
