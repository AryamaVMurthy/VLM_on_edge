//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <random>
#include <string>

#include "qualla/detail/basic-sampler.hpp"
#include "qualla/detail/config.hpp"
#include "qualla/detail/onload.hpp"
#include "qualla/detail/sampler-utils.hpp"
#include "qualla/sampler.hpp"

#define __INFO(__fmt, ...)  _env.logger().post(Logger::INFO, fmt::format(__fmt, ##__VA_ARGS__))
#define __WARN(__fmt, ...)  _env.logger().post(Logger::WARN, fmt::format(__fmt, ##__VA_ARGS__))
#define __ERROR(__fmt, ...) _env.logger().post(Logger::ERROR, fmt::format(__fmt, ##__VA_ARGS__))
#define __KPIS(__fmt, ...) \
  _env.logger().post(Logger::SAMPLER_KPIS, [&]() { return fmt::format(__fmt, ##__VA_ARGS__); })
#define __DEBUG(__fmt, ...) \
  _env.logger().post(Logger::SAMPLER_DEBUG, [&]() { return fmt::format(__fmt, ##__VA_ARGS__); })
#define __TRACE(__fmt, ...) \
  _env.logger().post(Logger::SAMPLER_TRACE, [&]() { return fmt::format(__fmt, ##__VA_ARGS__); })

namespace fs = std::filesystem;

namespace qualla {

BasicSampler::BasicSampler(Context& ctx, const json& conf) : Sampler(ctx, "basic", conf) {
  // Parse config
  using qc = qualla::Config;
  // Init rng
  _temp   = qc::optional<float>(conf, "temp", 0.1);
  _top_k  = qc::optional<size_t>(conf, "top-k", 0);
  _top_p  = qc::optional<float>(conf, "top-p", 0.8);
  _greedy = (_temp <= 0.f || _top_k == 1);
  _rng.seed(_seed != -1 ? _seed : std::time(nullptr));
}

int32_t BasicSampler::_process(std::span<const float> logits,
                               std::vector<float>* probs_out,
                               bool tok_out) {
  const size_t n_vocab = _ctx.n_vocab();

  assert(logits.size() % n_vocab == 0);
  assert(logits.size() / n_vocab == 1);

  const float temp    = _temp;
  const int32_t top_k = _top_k <= 0 ? n_vocab : _top_k;
  const float top_p   = _top_p;

  __DEBUG("input-logits: {} ... {}", logits.first(10), logits.last(10));

  IndexedLogits indexed_logits(logits, _rng);

  int32_t id = -1;

  if (_greedy) {
    // Greedy sampling
    id = indexed_logits.sampleGreedyUnsorted();
  } else {
    // Temperature sampling
    if (top_k > 0) {
      indexed_logits.topK(top_k);
    }

    indexed_logits.topP(top_p, 1);

    if (_gumbel) {
      indexed_logits.logSoftmax(temp);
      id = tok_out ? indexed_logits.sampleUsingGumbelMax() : -1;
    } else {
      indexed_logits.softmax(temp);
      id = tok_out ? indexed_logits.sampleFromProbs() : -1;
    }
  }

  // Output probability distribution
  if (probs_out) {
    QUALLA_ASSERT(indexed_logits.probs_valid);

    // Expand the output vector and fill it with the default values
    probs_out->resize(probs_out->size() + n_vocab,
                      _gumbel ? -std::numeric_limits<float>::infinity() : 0);

    auto p = std::span(probs_out->data(), probs_out->size()).last(n_vocab);
    for (size_t i = 0; i < indexed_logits.size(); i++) {
      int t = (int)indexed_logits.indices[i];
      p[t]  = indexed_logits.probs[i];
    }
  }

  _env.logger().compose(Logger::SAMPLER_DEBUG, [&](Logger::Helper w) {
    int32_t N = 5;
    // FIXME: This sort over-here is disruptive to the output
    // indexed_logits.sort(N);

    const auto& I{indexed_logits};
    w.write(fmt::format(
        "top-{} tokens: {}", N, std::span{I.indices.data(), I.indices.size()}.first(N)));
    w.write(
        fmt::format("top-{} logits: {}", N, std::span{I.logits.data(), I.logits.size()}.first(N)));
    w.write(
        fmt::format("top-{} probs:  {}", N, std::span{I.probs.data(), I.probs.size()}.first(N)));
  });

  return id;
}

int32_t BasicSampler::process(std::span<const float> logits) {
  return _process(logits, nullptr, true);
}

int32_t BasicSampler::process(std::span<const float> logits,
                              std::vector<float>& probs_out,
                              bool tok_out) {
  return _process(logits, &probs_out, tok_out);
}

// return multiple tokens - top_k after processing, temperature, top_p, gumbel, etc
std::vector<int32_t> BasicSampler::process(std::span<const float>& logits,
                                           std::vector<float>& probs,
                                           int32_t num_return) {
  const size_t n_vocab = _ctx.n_vocab();

  assert(logits.size() % n_vocab == 0);
  assert(logits.size() / n_vocab == 1);

  const float temp  = _temp;
  const float top_p = _top_p;
  num_return        = num_return <= 0 ? n_vocab : num_return;

  __DEBUG("input-logits: {} ... {}", logits.first(10), logits.last(10));

  IndexedLogits indexed_logits(logits, _rng);

  std::vector<int32_t> ids;

  // Temperature sampling
  indexed_logits.topP(top_p, 1);
  // add gumbel noise to the logits
  if (_gumbel) {
    indexed_logits.logSoftmax(temp);
    indexed_logits.addGumbelNoise();
  } else {
    indexed_logits.softmax(temp);
  }

  num_return =
      num_return <= indexed_logits.indices.size() ? num_return : indexed_logits.indices.size();
  indexed_logits.topK(num_return);
  ids = indexed_logits.indices;
  for (int i = 0; i < indexed_logits.probs.size(); i++) {
    probs[i] = indexed_logits.probs[i];
  }

  _env.logger().compose(Logger::SAMPLER_DEBUG, [&](Logger::Helper w) {
    int32_t N = 5;
    // FIXME: This sort over-here is disruptive to the output
    // indexed_logits.sort(N);

    const auto& I{indexed_logits};
    w.write(fmt::format(
        "top-{} tokens: {}", N, std::span{I.indices.data(), I.indices.size()}.first(N)));
    w.write(
        fmt::format("top-{} logits: {}", N, std::span{I.logits.data(), I.logits.size()}.first(N)));
    w.write(
        fmt::format("top-{} probs:  {}", N, std::span{I.probs.data(), I.probs.size()}.first(N)));
  });

  return ids;
}

bool BasicSampler::save(const std::string& name) {
  fs::path save_path = std::filesystem::path(name) / fmt::format("sampler.{}.rng", _role);

  std::fstream f(save_path, std::ios::out | std::ios::trunc);
  if (!f.is_open()) {
    __ERROR("basic-sampler: failed to open {} for writing", save_path.string());
    return false;
  }

  f << _rng;
  f.close();

  return true;
}

bool BasicSampler::restore(const std::string& name) {
  fs::path restore_path = std::filesystem::path(name) / fmt::format("sampler.{}.rng", _role);

  std::fstream f(restore_path, std::ios::in);
  if (!f.is_open()) {
    __ERROR("basic-sampler: failed to open {} for reading", restore_path.string());
    return false;
  }

  f >> _rng;
  f.close();

  return true;
}

void BasicSampler::reset() {
  // Just need to reinit rng
  _rng.seed(_seed);
}

// Registrator instance
static OnLoad regy([]() {
  Sampler::__register("basic", [](Context& ctx, const json& conf) {
    return (Sampler*)new BasicSampler(ctx, conf);
  });
});

void needBasicSampler() {}

void BasicSampler::applyConfig(const json& conf) {
  if (conf.contains("seed")) _seed = conf["seed"];
  if (conf.contains("temp")) _temp = conf["temp"];

  if (conf.contains("top-k")) _top_k = conf["top-k"];
  if (conf.contains("top-p")) _top_p = conf["top-p"];
}

}  // namespace qualla
