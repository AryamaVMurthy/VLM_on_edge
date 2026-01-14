//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <unordered_map>

#include "qualla/detail/config.hpp"
#include "qualla/detail/timer.hpp"
#include "qualla/embedding.hpp"
#include "qualla/logger.hpp"

namespace fs = std::filesystem;

namespace qualla {

Embedding::Embedding(std::shared_ptr<Env> env, const std::string& name, const qualla::json& json)
    : _name(name), _env(env) {
  Timer start;

  _env->logger().debug(fmt::format("embedding-new: {} config {}", name, json.dump()));

  using qc = qualla::Config;

  // Parse prompt config
  const qualla::json& pmt_conf = qc::optional<qualla::json>(json, "prompt", {});
  _tags                        = qc::optional<std::vector<std::string>>(pmt_conf, "tags", {"", ""});

  // Create the context first
  _ctx = Context::create(*_env, name, qc::optional<qualla::json>(json, "context", {}));

  // Create Tokenizer
  fs::path tok_path = _env->path().models / qc::mandatory<std::string>(json, "tokenizer");
  _tokenizer        = Tokenizer::create(*_ctx, tok_path);

  // Create Engine
  const qualla::json& eng_conf = qc::mandatory<qualla::json>(json, "engine");
  _engine                      = Engine::create(*_ctx, eng_conf);

  // Truncation of input to context
  _input_truncation = qc::optional<qualla::json>(json, "truncate-input", false);

  using FF = Engine::Feature::Flags;
  if (!_engine->supports(FF::OUTPUT_EMBEDDINGS))
    throw std::runtime_error("engine must output embeddings");

  _kpis.init.update(start.elapsed_usec());
}

Embedding::~Embedding() {}

bool Embedding::process(std::vector<int32_t>& tokens, std::vector<float>& output) {
  Timer start;

  State::clear();

  size_t n = _engine->process(tokens, output, false);
  if (!n) {
    State::error("engine prompt processing failed");
    return false;
  }

  _n_prompt += tokens.size();

  // Clean the buffer before using
  _output_dimensions.clear();

  uint64_t output_size = 1;
  // push number of tokens present in the result.
  _output_dimensions.push_back(n);
  // push back the dimension of the each embedding
  _output_dimensions.push_back(_ctx->n_embd());

  output_size = n * _ctx->n_embd();

  output.resize(output_size);

  _kpis.prompt.update(start.elapsed_usec());

  // Log latest KPIs in a single line
  _env->logger().post(Logger::KPIS, kpis().dump(" "));

  return true;
}

bool Embedding::query(const std::string& str, std::vector<float>& output) {
  std::string p_str;           // prompt string
  std::vector<int32_t> p_vec;  // prompt tokens

  p_vec.reserve(_ctx->n_ctx());

  p_str = _tags[0] + str + _tags[1];

  _env->logger().debug(fmt::format("embedding-query: {}", str));
  _env->logger().debug(fmt::format("embedding-prompt: {}", p_str));

  _n_queries++;

  _tokenizer->encode(p_str, p_vec);

  _env->logger().debug(fmt::format("embedding-tokens: {}", p_vec));

  if (p_vec.size() > (_ctx->n_ctx())) {  // Condition to not allow input to exceed context.
    if (_input_truncation == false) {
      throw std::runtime_error("Input exceeds the context of the model.");
    } else {
      p_vec.resize(_ctx->n_ctx());
    }
  }

  return process(p_vec, output);
}

// Embedding KPIs helpers

void Embedding::output_dimensions(std::vector<std::uint32_t>& outputDimensions) {
  outputDimensions = _output_dimensions;
}

// Get latest KPIs
Embedding::KPIs& Embedding::kpis() {
  // Update TPS
  if (_n_prompt) {
    float t            = _kpis.prompt.total_usec / _n_prompt;
    _kpis.tps.n_prompt = _n_prompt;
    _kpis.tps.prompt   = 1000000.0 / (t ? t : 1000000.0);
  }

  // We could synthesize more KPIs from from other layers (engine, sampler, etc)
  return _kpis;
}

std::string Embedding::KPIs::dump(std::string_view sep) const {
  return fmt::format("init:[{}]{}prompt:[{}]{} tps-prompt:{:.2f}",
                     init.dump(),
                     sep,
                     prompt.dump(),
                     sep,
                     tps.prompt);
}

void Embedding::KPIs::reset() {
  init.reset();
  prompt.reset();
  tps.prompt = 0.0;
}

// Create API

std::unique_ptr<Embedding> Embedding::create(std::shared_ptr<Env> env,
                                             const std::string& name,
                                             const qualla::json& conf) {
  return std::make_unique<Embedding>(env, name, conf);
}

std::unique_ptr<Embedding> Embedding::create(std::shared_ptr<Env> env,
                                             const std::string& name,
                                             std::istream& json_stream) {
  return create(env, name, json::parse(json_stream));
}

std::unique_ptr<Embedding> Embedding::create(std::shared_ptr<Env> env,
                                             const std::string& name,
                                             const fs::path& json_path) {
  if (!fs::exists(json_path))
    throw std::runtime_error(json_path.string() + ": file does not exist");
  std::ifstream ifs(json_path);
  return create(env, name, ifs);
}

}  // namespace qualla
