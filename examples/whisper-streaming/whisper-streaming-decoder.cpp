#include "whisper-streaming-decoder.h"

#include <algorithm>
#include <stdexcept>
#include <thread>

// The Whisper encoder maps 30 seconds of audio to 1500 mel-time frames.
static constexpr int WHISPER_MAX_AUDIO_CTX = 1500;
static constexpr int WHISPER_CHUNK_SAMPLES = 30 * WHISPER_SAMPLE_RATE;

WhisperStreamingDecoder::WhisperStreamingDecoder(const std::string &model_path)
{
  whisper_context_params cparams = whisper_context_default_params();
  cparams.use_gpu = true;

  ctx = whisper_init_from_file_with_params_no_state(model_path.c_str(), cparams);
  if (!ctx)
  {
    throw std::runtime_error("Failed to load whisper model: " + model_path);
  }

  state = whisper_init_state(ctx);
  if (!state)
  {
    whisper_free(ctx);
    ctx = nullptr;
    throw std::runtime_error("Failed to initialize whisper state for model: " + model_path);
  }
}

WhisperStreamingDecoder::~WhisperStreamingDecoder()
{
  if (state)
  {
    whisper_free_state(state);
    state = nullptr;
  }
  if (ctx)
  {
    whisper_free(ctx);
    ctx = nullptr;
  }
}

whisper_full_params WhisperStreamingDecoder::get_base_params()
{
  whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

  params.n_threads = std::min(4, (int)std::thread::hardware_concurrency());
  params.language = "en";
  params.no_context = true; // we manage context ourselves via prompt_tokens
  params.single_segment = false;
  params.print_progress = false;
  params.print_realtime = false;
  params.print_timestamps = false;

  // Inject the rolling context from previous utterances as a prompt.
  // prompt_tokens points into our stable vector; the pointer remains valid
  // for the synchronous duration of the whisper_full_* call below.
  if (!context_tokens.empty())
  {
    params.prompt_tokens = context_tokens.data();
    params.prompt_n_tokens = (int)context_tokens.size();
  }

  return params;
}

void WhisperStreamingDecoder::update_context_history()
{
  const int n_seg = whisper_full_n_segments_from_state(state);
  for (int i = 0; i < n_seg; ++i)
  {
    const int n_tok = whisper_full_n_tokens_from_state(state, i);
    for (int j = 0; j < n_tok; ++j)
    {
      context_tokens.push_back(whisper_full_get_token_id_from_state(state, i, j));
    }
  }
  if (context_tokens.size() > MAX_CONTEXT_TOKENS)
  {
    context_tokens.erase(
        context_tokens.begin(),
        context_tokens.begin() + (int)(context_tokens.size() - MAX_CONTEXT_TOKENS));
  }
}

std::string WhisperStreamingDecoder::decode_utterance(const std::vector<float> &pcmf32_samples)
{
  if (pcmf32_samples.empty())
  {
    return "";
  }

  std::string result;

  if ((int)pcmf32_samples.size() <= WHISPER_CHUNK_SAMPLES)
  {
    // ── Short path (≤ 30 s) ──────────────────────────────────────────────
    // Trim the encoder's audio context window to the actual utterance length
    // so we don't pay for padding the full 30-second window.
    whisper_full_params params = get_base_params();

    const int required_ctx = (int)((pcmf32_samples.size() / (float)WHISPER_SAMPLE_RATE) *
                                   (WHISPER_MAX_AUDIO_CTX / 30.0f)) +
                             128;
    params.audio_ctx = std::min(required_ctx, WHISPER_MAX_AUDIO_CTX);

    if (whisper_full_with_state(ctx, state, params,
                                pcmf32_samples.data(), (int)pcmf32_samples.size()) != 0)
    {
      return "";
    }

    const int n_seg = whisper_full_n_segments_from_state(state);
    for (int i = 0; i < n_seg; ++i)
    {
      result += whisper_full_get_segment_text_from_state(state, i);
    }

    update_context_history();
  }
  else
  {
    // ── Long path (> 30 s) ───────────────────────────────────────────────
    // whisper_full_parallel internally slices the audio into 30-second windows
    // and processes them concurrently, storing results in ctx's default state.
    whisper_full_params params = get_base_params();

    const int n_proc = std::max(1, (int)std::thread::hardware_concurrency() / 2);
    if (whisper_full_parallel(ctx, params,
                              pcmf32_samples.data(), (int)pcmf32_samples.size(),
                              n_proc) != 0)
    {
      return "";
    }

    // Results are in ctx's default state — use the non-_from_state accessors.
    const int n_seg = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_seg; ++i)
    {
      result += whisper_full_get_segment_text(ctx, i);
    }

    // Update context token history from the default state.
    for (int i = 0; i < n_seg; ++i)
    {
      const int n_tok = whisper_full_n_tokens(ctx, i);
      for (int j = 0; j < n_tok; ++j)
      {
        context_tokens.push_back(whisper_full_get_token_id(ctx, i, j));
      }
    }
    if (context_tokens.size() > MAX_CONTEXT_TOKENS)
    {
      context_tokens.erase(
          context_tokens.begin(),
          context_tokens.begin() + (int)(context_tokens.size() - MAX_CONTEXT_TOKENS));
    }
  }

  return result;
}

void WhisperStreamingDecoder::reset_context(const std::string &initial_prompt)
{
  context_tokens.clear();
  if (!initial_prompt.empty())
  {
    context_tokens.resize(initial_prompt.size());
    int n_tokens = whisper_tokenize(ctx, initial_prompt.c_str(), context_tokens.data(), (int)context_tokens.size());
    if (n_tokens < 0)
    {
      context_tokens.resize(-n_tokens);
      n_tokens = whisper_tokenize(ctx, initial_prompt.c_str(), context_tokens.data(), (int)context_tokens.size());
    }
    context_tokens.resize(std::max(0, n_tokens));
  }
}
