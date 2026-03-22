#include "common.h"
#include "common-whisper.h"

#include "whisper.h"
#include "httplib.h"
#include "json.hpp"

#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <memory>
#include <csignal>
#include <atomic>
#include <functional>
#include <cstdlib>
#include <mutex>
#include <condition_variable>
#include <optional>

#if defined (_WIN32)
#include <windows.h>
#endif

using namespace httplib;
using json = nlohmann::ordered_json;

namespace {

struct server_params {
    std::string hostname = "127.0.0.1";
    std::string inference_path = "/inference";
    int32_t port          = 8080;
};

struct whisper_params {
    int32_t n_threads     = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t n_processors  = 1;
    float   vad_threshold = 0.4f;
    std::string language  = "en";
    std::string model     = "models/ggml-base.en.bin";
    std::string vad_model = "";
};

class VadSegmenterGenerator {
public:
    struct Params {
        std::string vad_model_path;
        float threshold = 0.4f;
        int min_speech_ms = 250;
        int min_silence_ms = 700;
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
        history_samples = 2.0f * sample_rate;
        pcm_history.resize(history_samples, 0.0f);
    }

    ~VadSegmenterGenerator() {
        if (vctx) whisper_vad_free(vctx);
    }

    std::vector<std::vector<float>> feed(const std::vector<float>& pcm_chunk) {
        std::vector<std::vector<float>> results;
        for (float s : pcm_chunk) {
            pcm_history[history_idx] = s;
            history_idx = (history_idx + 1) % history_samples;
            if (state == State::IN_SPEECH) pcm_buffer.push_back(s);
            samples_since_vad++;
            if (samples_since_vad >= n_window) {
                samples_since_vad = 0;
                bool is_speech = run_vad();
                if (state == State::SEARCHING) {
                    if (is_speech) {
                        speech_frames++;
                        if ((params.min_speech_ms * sample_rate) / 1000 <= speech_frames * n_window) {
                            state = State::IN_SPEECH;
                            pcm_buffer.clear();
                            int n_pre = padding_samples;
                            for (int i = 0; i < n_pre; ++i) {
                                int idx = (history_idx - n_pre + i + history_samples) % history_samples;
                                pcm_buffer.push_back(pcm_history[idx]);
                            }
                            silence_frames = 0;
                        }
                    } else speech_frames = 0;
                } else {
                    if (!is_speech) {
                        silence_frames++;
                        if (silence_frames * n_window >= min_silence_samples) {
                            state = State::SEARCHING;
                            speech_frames = 0;
                            results.push_back(std::move(pcm_buffer));
                            pcm_buffer.clear();
                        }
                    } else silence_frames = 0;
                }
            }
        }
        return results;
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

    enum class State { SEARCHING, IN_SPEECH };
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

struct stream_state {
    std::vector<uint8_t> byte_buffer;
    std::vector<std::string> results;
    std::mutex mutex;
    std::condition_variable cv;
    std::unique_ptr<VadSegmenterGenerator> segmenter;
    bool is_finished = false;
};

json get_result_json(struct whisper_context * ctx, const whisper_params & params) {
    const int n_segments = whisper_full_n_segments(ctx);
    json jres = json{
        {"language", whisper_lang_str_full(whisper_full_lang_id(ctx))},
        {"segments", json::array()}
    };
    std::string results = "";
    for (int i = 0; i < n_segments; ++i) {
        const char * text = whisper_full_get_segment_text(ctx, i);
        results += text;
        json segment = json{
            {"id", i},
            {"text", text},
            {"start", (float)whisper_full_get_segment_t0(ctx, i) * 0.01f},
            {"end", (float)whisper_full_get_segment_t1(ctx, i) * 0.01f},
        };
        jres["segments"].push_back(segment);
    }
    jres["text"] = results;
    return jres;
}

bool whisper_params_parse(int argc, char ** argv, whisper_params & params, server_params & sparams) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-m" || arg == "--model") params.model = argv[++i];
        else if (arg == "-vm" || arg == "--vad-model") params.vad_model = argv[++i];
        else if (arg == "--port") sparams.port = std::stoi(argv[++i]);
        else if (arg == "--host") sparams.hostname = argv[++i];
        else if (arg == "-t" || arg == "--threads") params.n_threads = std::stoi(argv[++i]);
    }
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    server_params sparams;
    whisper_params default_params;
    if (!whisper_params_parse(argc, argv, default_params, sparams)) return 1;

    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = true;
    cparams.flash_attn = true;

    struct whisper_context * ctx = whisper_init_from_file_with_params(default_params.model.c_str(), cparams);
    if (!ctx) return 1;

    std::mutex whisper_mutex;
    Server svr;

    svr.Post("/inference", [&](const Request &req, Response &res, const ContentReader &reader) {
        whisper_params params = default_params;
        bool is_sse = req.get_header_value("Accept") == "text/event-stream";

        if (is_sse) {
            fprintf(stderr, "SSE: New connection\n");
            auto sstate = std::make_shared<stream_state>();
            if (!params.vad_model.empty()) {
                VadSegmenterGenerator::Params vparams;
                vparams.vad_model_path = params.vad_model;
                sstate->segmenter = std::make_unique<VadSegmenterGenerator>(vparams);
            }

            res.set_chunked_content_provider("text/event-stream", [sstate](size_t offset, DataSink &sink) {
                std::unique_lock<std::mutex> lock(sstate->mutex);
                if (sstate->results.empty() && !sstate->is_finished) {
                    sstate->cv.wait_for(lock, std::chrono::milliseconds(100), [&]{ return !sstate->results.empty() || sstate->is_finished; });
                }
                for (const auto & r : sstate->results) {
                    if (!sink.write(r.c_str(), r.size())) return false;
                }
                sstate->results.clear();
                if (sstate->is_finished) {
                    sink.done();
                    return false;
                }
                return true;
            });

            std::thread reader_thread([&whisper_mutex, ctx, params, sstate, reader]() {
                size_t total_bytes_received = 0;
                reader([&](const char *data, size_t data_len) {
                    total_bytes_received += data_len;
                    std::vector<float> pcm_chunk;
                    {
                        std::lock_guard<std::mutex> lock(sstate->mutex);
                        sstate->byte_buffer.insert(sstate->byte_buffer.end(), (const uint8_t *)data, (const uint8_t *)data + data_len);
                        size_t offset = 0;
                        if (total_bytes_received <= data_len + 44 && sstate->byte_buffer.size() >= 44) {
                            if (memcmp(sstate->byte_buffer.data(), "RIFF", 4) == 0) offset = 44;
                        }
                        size_t n_samples = (sstate->byte_buffer.size() - offset) / 2;
                        if (n_samples > 0) {
                            pcm_chunk.reserve(n_samples);
                            const int16_t * raw = (const int16_t *)(sstate->byte_buffer.data() + offset);
                            for (size_t i = 0; i < n_samples; i++) pcm_chunk.push_back(float(raw[i]) / 32768.0f);
                            size_t consumed = offset + (n_samples * 2);
                            std::move(sstate->byte_buffer.begin() + consumed, sstate->byte_buffer.end(), sstate->byte_buffer.begin());
                            sstate->byte_buffer.resize(sstate->byte_buffer.size() - consumed);
                        }
                    }

                    if (sstate->segmenter) {
                        auto segments = sstate->segmenter->feed(pcm_chunk);
                        for (const auto& speech_seg : segments) {
                            auto t_start = std::chrono::high_resolution_clock::now();
                            std::lock_guard<std::mutex> lock(whisper_mutex);
                            whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
                            wparams.n_threads = params.n_threads;
                            // Match CLI's parallel strategy
                            if (whisper_full_parallel(ctx, wparams, speech_seg.data(), speech_seg.size(), 1) == 0) {
                                json jres = get_result_json(ctx, params);
                                auto t_end = std::chrono::high_resolution_clock::now();
                                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
                                fprintf(stderr, "SSE: Inference took %ld ms for %.2f s audio\n", ms, (float)speech_seg.size()/16000.0f);
                                std::lock_guard<std::mutex> slock(sstate->mutex);
                                sstate->results.push_back("data: " + jres.dump() + "\n\n");
                                sstate->cv.notify_all();
                            }
                        }
                    }
                    return true;
                });

                if (sstate->segmenter) {
                    if (auto speech_seg = sstate->segmenter->flush()) {
                        std::lock_guard<std::mutex> lock(whisper_mutex);
                        whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
                        wparams.n_threads = params.n_threads;
                        if (whisper_full_parallel(ctx, wparams, speech_seg->data(), speech_seg->size(), 1) == 0) {
                            json jres = get_result_json(ctx, params);
                            std::lock_guard<std::mutex> slock(sstate->mutex);
                            sstate->results.push_back("data: " + jres.dump() + "\n\n");
                            sstate->cv.notify_all();
                        }
                    }
                }
                {
                    std::lock_guard<std::mutex> slock(sstate->mutex);
                    sstate->is_finished = true;
                    sstate->cv.notify_all();
                }
            });
            reader_thread.detach();
            return;
        }

        std::string body;
        reader([&](const char *data, size_t data_len) { body.append(data, data_len); return true; });
        std::vector<float> pcmf32;
        std::vector<std::vector<float>> pcmf32s;
        if (::read_audio_data(body, pcmf32, pcmf32s, false)) {
            std::lock_guard<std::mutex> lock(whisper_mutex);
            whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
            wparams.n_threads = params.n_threads;
            whisper_full_parallel(ctx, wparams, pcmf32.data(), pcmf32.size(), 1);
            res.set_content(get_result_json(ctx, params).dump(), "application/json");
        }
    });

    fprintf(stderr, "Whisper server listening at http://%s:%d\n", sparams.hostname.c_str(), sparams.port);
    svr.listen(sparams.hostname.c_str(), sparams.port);
    whisper_free(ctx);
    return 0;
}
