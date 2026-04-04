#pragma once

#include <string>
#include <vector>
#include "whisper.h"

class WhisperStreamingDecoder
{
public:
  // Initializes the Whisper context and allocates the persistent state workspace
  explicit WhisperStreamingDecoder(const std::string &model_path);

  // Cleans up Whisper context and state resources
  ~WhisperStreamingDecoder();

  // Delete copy constructors to prevent double-freeing the underlying C pointers
  WhisperStreamingDecoder(const WhisperStreamingDecoder &) = delete;
  WhisperStreamingDecoder &operator=(const WhisperStreamingDecoder &) = delete;

  // Core Interface:
  // Receives a discrete audio segment (16kHz PCM), routes it to the optimal
  // internal decoding path based on length, and returns the decoded text.
  // Automatically applies previous utterance context to the decoder.
  std::string decode_utterance(const std::vector<float> &pcmf32_samples);

  // Flushes the internal token history.
  // Call this when a conversation ends or you want to start a completely fresh context.
  // If initial_prompt is provided, it will be tokenized and used as the starting context.
  void reset_context(const std::string &initial_prompt = "");

private:
  whisper_context *ctx;
  whisper_state *state;

  // Persistent buffer of recently decoded tokens.
  // This is passed as 'prompt_tokens' to the next inference run to maintain continuity.
  std::vector<whisper_token> context_tokens;

  // Hard limit to prevent the prompt array from growing indefinitely over a long session
  const size_t MAX_CONTEXT_TOKENS = 256;

  // Internal helper to extract tokens from the latest successful decode (short path)
  // and append them to the 'context_tokens' buffer, applying the MAX_CONTEXT_TOKENS ceiling.
  void update_context_history();

  // Internal helper for configuring baseline params (greedy decoding, no internal context)
  whisper_full_params get_base_params();
};
