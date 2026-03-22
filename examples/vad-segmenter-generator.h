#pragma once

#include <vector>
#include <optional>
#include <string>
#include <memory>

class VadSegmenterGenerator {
public:
    struct Params {
        std::string vad_model_path;
        float threshold = 0.5f;
        int min_speech_ms = 250;
        int min_silence_ms = 500;
        int padding_ms = 300;
    };

    VadSegmenterGenerator(const Params& params);
    ~VadSegmenterGenerator();

    // Prevent copying due to unique_ptr
    VadSegmenterGenerator(const VadSegmenterGenerator&) = delete;
    VadSegmenterGenerator& operator=(const VadSegmenterGenerator&) = delete;

    std::optional<std::vector<float>> feed(const std::vector<float>& pcm_chunk);
    std::optional<std::vector<float>> flush();

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};
