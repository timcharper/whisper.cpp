#include "common.h"
#include "common-whisper.h"
#include "vad-segmenter-generator.h"
#include "whisper-streaming-decoder.h"

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

#if defined(_WIN32)
#include <windows.h>
#endif

using namespace httplib;
using json = nlohmann::ordered_json;

namespace
{

    struct server_params
    {
        std::string hostname = "127.0.0.1";
        std::string inference_path = "/inference";
        int32_t port = 8080;
    };

    struct whisper_params
    {
        int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());
        int32_t n_processors = 1;
        float vad_threshold = 0.4f;
        std::string language = "en";
        std::string model = "models/ggml-base.en.bin";
        std::string vad_model = "";
    };

    struct stream_state
    {
        std::vector<uint8_t> byte_buffer;
        std::vector<std::string> results;
        std::mutex mutex;
        std::condition_variable cv;
        std::unique_ptr<VadSegmenterGenerator> segmenter;
        bool is_finished = false;
    };

    json get_result_json(struct whisper_context *ctx, const whisper_params &params)
    {
        const int n_segments = whisper_full_n_segments(ctx);
        json jres = json{
            {"language", whisper_lang_str_full(whisper_full_lang_id(ctx))},
            {"segments", json::array()}};
        std::string results = "";
        for (int i = 0; i < n_segments; ++i)
        {
            const char *text = whisper_full_get_segment_text(ctx, i);
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

    bool whisper_params_parse(int argc, char **argv, whisper_params &params, server_params &sparams)
    {
        for (int i = 1; i < argc; i++)
        {
            std::string arg = argv[i];
            if (arg == "-m" || arg == "--model")
                params.model = argv[++i];
            else if (arg == "-vm" || arg == "--vad-model")
                params.vad_model = argv[++i];
            else if (arg == "--port")
                sparams.port = std::stoi(argv[++i]);
            else if (arg == "--host")
                sparams.hostname = argv[++i];
            else if (arg == "-t" || arg == "--threads")
                params.n_threads = std::stoi(argv[++i]);
        }
        return true;
    }

} // namespace

int main(int argc, char **argv)
{
    server_params sparams;
    whisper_params default_params;
    if (!whisper_params_parse(argc, argv, default_params, sparams))
        return 1;

    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = true;
    cparams.flash_attn = true;

    struct whisper_context *ctx = whisper_init_from_file_with_params(default_params.model.c_str(), cparams);
    if (!ctx)
        return 1;

    // Stateful streaming decoder: owns its own context + state + rolling token history.
    // Used exclusively by the SSE streaming path.
    WhisperStreamingDecoder stream_decoder(default_params.model);

    std::mutex whisper_mutex;
    Server svr;

    svr.Post("/inference", [&](const Request &req, Response &res, const ContentReader &reader)
             {
        whisper_params params = default_params;
        bool is_sse = req.get_header_value("Accept") == "text/event-stream";

        if (req.has_param("prompt")) {
            params.prompt = req.get_param_value("prompt");
        }

        if (is_sse) {
            fprintf(stderr, "SSE: New connection (prompt: '%s')\n", params.prompt.c_str());
            auto sstate = std::make_shared<stream_state>();
            // Each SSE connection starts with a fresh transcript context, seeded by the optional prompt.
            stream_decoder.reset_context(params.prompt);
            if (!params.vad_model.empty()) {
                VadSegmenterGenerator::Params vparams;
                vparams.vad_model_path = params.vad_model;
                vparams.threshold      = params.vad_threshold;
                vparams.min_silence_ms = 700;
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

            std::thread reader_thread([&whisper_mutex, &stream_decoder, params, sstate, reader]() {
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
                        if (auto speech_seg = sstate->segmenter->feed(pcm_chunk)) {
                            auto t_start = std::chrono::high_resolution_clock::now();
                            std::string text;
                            {
                                std::lock_guard<std::mutex> lock(whisper_mutex);
                                text = stream_decoder.decode_utterance(*speech_seg);
                            }
                            auto t_end = std::chrono::high_resolution_clock::now();
                            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
                            fprintf(stderr, "SSE: Inference took %ld ms for %.2f s audio\n",
                                    ms, (float)speech_seg->size() / 16000.0f);
                            if (!text.empty()) {
                                std::lock_guard<std::mutex> slock(sstate->mutex);
                                sstate->results.push_back("data: " + json{{"text", text}}.dump() + "\n\n");
                                sstate->cv.notify_all();
                            }
                        }
                    }
                    return true;
                });

                if (sstate->segmenter) {
                    if (auto speech_seg = sstate->segmenter->flush()) {
                        std::string text;
                        {
                            std::lock_guard<std::mutex> lock(whisper_mutex);
                            text = stream_decoder.decode_utterance(*speech_seg);
                        }
                        if (!text.empty()) {
                            std::lock_guard<std::mutex> slock(sstate->mutex);
                            sstate->results.push_back("data: " + json{{"text", text}}.dump() + "\n\n");
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
            wparams.initial_prompt = params.prompt.c_str();
            whisper_full_parallel(ctx, wparams, pcmf32.data(), pcmf32.size(), 1);
            res.set_content(get_result_json(ctx, params).dump(), "application/json");
        } });

    fprintf(stderr, "Whisper server listening at http://%s:%d\n", sparams.hostname.c_str(), sparams.port);
    svr.listen(sparams.hostname.c_str(), sparams.port);
    whisper_free(ctx);
    return 0;
}
