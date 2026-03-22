#include "vad-segmenter-generator.h"
#include "common-whisper.h"
#include "common.h"

#include <vector>
#include <cstdio>
#include <algorithm>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <vad_model_path> <audio_path>\n", argv[0]);
        return 1;
    }

    std::vector<float> pcmf32;
    std::vector<std::vector<float>> pcmf32s;
    if (!read_audio_data(argv[2], pcmf32, pcmf32s, false)) {
        fprintf(stderr, "Failed to read audio data from %s\n", argv[2]);
        return 1;
    }

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
    if (auto seg = gen.flush()) {
        printf("Speech Segment (flush): len %.2f\n", (float)seg->size()/16000.0f);
    }

    return 0;
}
