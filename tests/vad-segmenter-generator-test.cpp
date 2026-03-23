#include "vad-segmenter-generator.h"
#include "common-whisper.h"
#include "common.h"

#include <vector>
#include <cstdio>
#include <algorithm>
#include <string>

static void write_segment_wav(const std::vector<float> &seg, int idx, const std::string &out_dir)
{
    char filename[512];
    snprintf(filename, sizeof(filename), "%s/segment_%03d.wav", out_dir.c_str(), idx);
    wav_writer writer;
    if (!writer.open(filename, 16000, 16, 1))
    {
        fprintf(stderr, "Failed to open %s for writing\n", filename);
        return;
    }
    writer.write(seg.data(), seg.size());
    printf("  -> wrote %s (%.2f s)\n", filename, (float)seg.size() / 16000.0f);
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        fprintf(stderr, "Usage: %s <vad_model_path> <audio_path> [output_dir]\n", argv[0]);
        return 1;
    }

    std::string out_dir = argc >= 4 ? argv[3] : ".";

    std::vector<float> pcmf32;
    std::vector<std::vector<float>> pcmf32s;
    if (!read_audio_data(argv[2], pcmf32, pcmf32s, false))
    {
        fprintf(stderr, "Failed to read audio data from %s\n", argv[2]);
        return 1;
    }

    VadSegmenterGenerator::Params p;
    p.vad_model_path = argv[1];
    p.min_silence_ms = 700;
    p.padding_ms = 200;

    VadSegmenterGenerator gen(p);
    int seg_idx = 0;
    int chunk_size = 1600; // 100ms
    size_t sample_pos = 0;
    for (size_t i = 0; i < pcmf32.size(); i += chunk_size)
    {
        std::vector<float> chunk(pcmf32.begin() + i, pcmf32.begin() + std::min(pcmf32.size(), i + chunk_size));
        if (auto seg = gen.feed(chunk))
        {
            float end_s = (float)i / 16000.0f;
            float start_s = end_s - (float)seg->size() / 16000.0f;
            printf("Segment %d: %.2f s -> %.2f s (%.2f s)\n", seg_idx, start_s, end_s, (float)seg->size() / 16000.0f);
            write_segment_wav(*seg, seg_idx++, out_dir);
        }
        sample_pos = i;
    }
    if (auto seg = gen.flush())
    {
        float end_s = (float)sample_pos / 16000.0f;
        float start_s = end_s - (float)seg->size() / 16000.0f;
        printf("Segment %d (flush): %.2f s -> %.2f s (%.2f s)\n", seg_idx, start_s, end_s, (float)seg->size() / 16000.0f);
        write_segment_wav(*seg, seg_idx++, out_dir);
    }

    printf("Total segments: %d\n", seg_idx);
    return 0;
}
