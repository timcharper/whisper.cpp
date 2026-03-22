#include "whisper.h"
#include "common-whisper.h"
#include "common.h"

#include <vector>
#include <optional>
#include <memory>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <deque>

class VadSegmenterGenerator {
public:
    struct Params {
        std::string vad_model_path;
        float threshold = 0.5f;
        int min_speech_ms = 250;
        int min_silence_ms = 500;
        int padding_ms = 300;
    };

    VadSegmenterGenerator(const Params& params) : params(params) {
        struct whisper_vad_context_params ctx_params = whisper_vad_default_context_params();
        vctx = whisper_vad_init_from_file_with_params(params.vad_model_path.c_str(), ctx_params);
        if (!vctx) throw std::runtime_error("VAD init failed");

        sample_rate = WHISPER_SAMPLE_RATE;
        n_window = 512;
        padding_samples = (params.padding_ms * sample_rate) / 1000;
        min_silence_samples = (params.min_silence_ms * sample_rate) / 1000;
        
        // We will keep a history of the last 2 seconds for VAD context
        history_samples = 2.0f * sample_rate;
        pcm_history.resize(history_samples, 0.0f);
    }

    ~VadSegmenterGenerator() {
        if (vctx) whisper_vad_free(vctx);
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
                            // Add pre-roll
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

private:
    enum class State { SEARCHING, IN_SPEECH };
    
    int speech_samples_needed() { return (params.min_speech_ms * sample_rate) / 1000; }

    bool run_vad() {
        // Feed a larger window to VAD so LSTM can "warm up"
        // We use the last 1536 samples (3 frames)
        float window[1536];
        for (int i = 0; i < 1536; ++i) {
            int idx = (history_idx - 1536 + i + history_samples) % history_samples;
            window[i] = pcm_history[idx];
        }
        // whisper_vad_detect_speech returns true if ANY chunk in the buffer is speech.
        // Since we only care about the CURRENT chunk (the last 512), we check the prob.
        if (!whisper_vad_detect_speech(vctx, window, 1536)) return false;
        
        float* probs = whisper_vad_probs(vctx);
        int n = whisper_vad_n_probs(vctx);
        return probs[n-1] > params.threshold;
    }

    Params params;
    struct whisper_vad_context * vctx = nullptr;
    State state = State::SEARCHING;
    int sample_rate, n_window, padding_samples, min_silence_samples, history_samples;
    std::vector<float> pcm_history;
    std::vector<float> pcm_buffer;
    size_t history_idx = 0;
    int samples_since_vad = 0;
    int speech_frames = 0;
    int silence_frames = 0;
};

int main(int argc, char** argv) {
    if (argc < 3) return 1;
    std::vector<float> pcmf32;
    std::vector<std::vector<float>> pcmf32s;
    if (!read_audio_data(argv[2], pcmf32, pcmf32s, false)) return 1;

    VadSegmenterGenerator::Params p;
    p.vad_model_path = argv[1];
    p.min_silence_ms = 700; // Increased for better discrete speech detection
    p.padding_ms = 200;
    
    VadSegmenterGenerator gen(p);
    int chunk_size = 1600; // 100ms
    for (size_t i = 0; i < pcmf32.size(); i += chunk_size) {
        std::vector<float> chunk(pcmf32.begin() + i, pcmf32.begin() + std::min(pcmf32.size(), i + chunk_size));
        if (auto seg = gen.feed(chunk)) {
            printf("Speech Segment: %.2f to %.2f (len %.2f)\n", 
                (float)(i - seg->size())/16000.0f, (float)i/16000.0f, (float)seg->size()/16000.0f);
        }
    }
    if (auto seg = gen.flush()) printf("Speech Segment (flush): len %.2f\n", (float)seg->size()/16000.0f);
    return 0;
}
