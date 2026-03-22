// Test runner for WhisperStreamingDecoder.
//
// 1. Reads samples/discrete-speech.wav (16 kHz mono).
// 2. Segments the audio with VadSegmenterGenerator.
// 3. Feeds each utterance to decode_utterance() one at a time.
// 4. Logs per-utterance duration, decode latency and RTF.
// 5. Prints the full concatenated transcript and overall RTF.

#include "whisper-streaming-decoder.h"
#include "common-whisper.h"
#include "vad-segmenter-generator.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static constexpr int VAD_CHUNK_SAMPLES = 512; // samples fed to VAD per call

struct test_params
{
  std::string model_path = "models/ggml-base.en.bin";
  std::string vad_model_path = "models/ggml-silero-v5.1.2.bin";
  std::string audio_path = "samples/discrete-speech.wav";
  float vad_threshold = 0.5f;
  int vad_min_speech_ms = 250;
  int vad_min_silence_ms = 500;
  int vad_padding_ms = 300;
};

static void print_usage(const char *prog, const test_params &p)
{
  fprintf(stderr, "\nusage: %s [options]\n\n", prog);
  fprintf(stderr, "options:\n");
  fprintf(stderr, "  -m  PATH   whisper model path            [%s]\n", p.model_path.c_str());
  fprintf(stderr, "  -vm PATH   VAD model path                [%s]\n", p.vad_model_path.c_str());
  fprintf(stderr, "  -f  PATH   audio file (16 kHz mono WAV)  [%s]\n", p.audio_path.c_str());
  fprintf(stderr, "  -vt FLOAT  VAD speech threshold          [%.2f]\n", p.vad_threshold);
  fprintf(stderr, "  -h         show this help\n\n");
}

static bool parse_params(int argc, char **argv, test_params &p)
{
  for (int i = 1; i < argc; ++i)
  {
    std::string arg = argv[i];
    if (arg == "-h" || arg == "--help")
    {
      print_usage(argv[0], p);
      return false;
    }
    else if (arg == "-m" && i + 1 < argc)
    {
      p.model_path = argv[++i];
    }
    else if (arg == "-vm" && i + 1 < argc)
    {
      p.vad_model_path = argv[++i];
    }
    else if (arg == "-f" && i + 1 < argc)
    {
      p.audio_path = argv[++i];
    }
    else if (arg == "-vt" && i + 1 < argc)
    {
      p.vad_threshold = std::stof(argv[++i]);
    }
    else
    {
      fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
      print_usage(argv[0], p);
      return false;
    }
  }
  return true;
}

static void cb_log_disable(enum ggml_log_level, const char *, void *) {}

int main(int argc, char **argv)
{
  test_params params;
  if (!parse_params(argc, argv, params))
  {
    return 1;
  }

  // Suppress verbose whisper/ggml logs so the timing table is readable.
  whisper_log_set(cb_log_disable, nullptr);

  // ── Load audio ────────────────────────────────────────────────────────────
  std::vector<float> pcmf32;
  std::vector<std::vector<float>> pcmf32s; // unused (mono)
  if (!read_audio_data(params.audio_path, pcmf32, pcmf32s, /*stereo=*/false))
  {
    fprintf(stderr, "error: failed to read audio from '%s'\n", params.audio_path.c_str());
    return 1;
  }

  const double total_audio_sec = pcmf32.size() / (double)WHISPER_SAMPLE_RATE;
  fprintf(stdout, "Audio loaded : %s  (%.2f s, %zu samples)\n\n",
          params.audio_path.c_str(), total_audio_sec, pcmf32.size());

  // ── VAD segmentation ─────────────────────────────────────────────────────
  VadSegmenterGenerator::Params vad_p;
  vad_p.vad_model_path = params.vad_model_path;
  vad_p.threshold = params.vad_threshold;
  vad_p.min_speech_ms = params.vad_min_speech_ms;
  vad_p.min_silence_ms = params.vad_min_silence_ms;
  vad_p.padding_ms = params.vad_padding_ms;

  VadSegmenterGenerator segmenter(vad_p);

  std::vector<std::vector<float>> utterances;
  for (size_t i = 0; i < pcmf32.size(); i += VAD_CHUNK_SAMPLES)
  {
    const size_t end = std::min(i + (size_t)VAD_CHUNK_SAMPLES, pcmf32.size());
    std::vector<float> chunk(pcmf32.begin() + i, pcmf32.begin() + end);
    if (auto seg = segmenter.feed(chunk))
    {
      utterances.push_back(std::move(*seg));
    }
  }
  if (auto seg = segmenter.flush())
  {
    utterances.push_back(std::move(*seg));
  }

  fprintf(stdout, "VAD segments : %zu utterances\n\n", utterances.size());
  if (utterances.empty())
  {
    fprintf(stderr, "warning: no speech detected — check VAD model path and threshold.\n");
    return 1;
  }

  // ── Decoder init ─────────────────────────────────────────────────────────
  fprintf(stdout, "Loading model: %s\n\n", params.model_path.c_str());
  WhisperStreamingDecoder decoder(params.model_path);

  // ── Per-utterance decoding ────────────────────────────────────────────────
  //    Columns: index | duration (s) | decode (ms) | RTF | path | text
  fprintf(stdout, "%-4s  %-10s  %-11s  %-6s  %-5s  %s\n",
          "#", "Dur(s)", "Decode(ms)", "RTF", "Path", "Text");
  fprintf(stdout, "%s\n", std::string(90, '-').c_str());

  std::string full_transcript;
  double total_inference_ms = 0.0;

  for (size_t idx = 0; idx < utterances.size(); ++idx)
  {
    const auto &utt = utterances[idx];
    const double duration_sec = utt.size() / (double)WHISPER_SAMPLE_RATE;
    const char *path_label = (utt.size() > (size_t)(30 * WHISPER_SAMPLE_RATE))
                                 ? "long"
                                 : "short";

    const auto t0 = std::chrono::steady_clock::now();
    std::string text = decoder.decode_utterance(utt);
    const auto t1 = std::chrono::steady_clock::now();

    const double decode_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double rtf = decode_ms / (duration_sec * 1000.0);
    total_inference_ms += decode_ms;

    // Trim leading/trailing whitespace from decoded text for cleaner output.
    while (!text.empty() && text.front() == ' ')
      text.erase(text.begin());

    fprintf(stdout, "%-4zu  %-10.2f  %-11.1f  %-6.3f  %-5s  %s\n",
            idx, duration_sec, decode_ms, rtf, path_label, text.c_str());

    full_transcript += text;
  }

  // ── Summary ───────────────────────────────────────────────────────────────
  fprintf(stdout, "\n%s\n", std::string(90, '=').c_str());
  fprintf(stdout, "Transcript:\n%s\n", full_transcript.c_str());
  fprintf(stdout, "\n%s\n", std::string(90, '=').c_str());

  const double overall_rtf = total_inference_ms / (total_audio_sec * 1000.0);
  fprintf(stdout, "Utterances   : %zu\n", utterances.size());
  fprintf(stdout, "Audio total  : %.2f s\n", total_audio_sec);
  fprintf(stdout, "Decode total : %.1f ms (%.2f s)\n",
          total_inference_ms, total_inference_ms / 1000.0);
  fprintf(stdout, "Overall RTF  : %.4f  %s\n",
          overall_rtf, overall_rtf < 1.0 ? "(faster than real-time)" : "(slower than real-time)");
  fprintf(stdout, "\n");

  return 0;
}
