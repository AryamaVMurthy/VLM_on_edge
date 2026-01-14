//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef QUALLA_DETAIL_BASIC_SAMPLER_HPP
#define QUALLA_DETAIL_BASIC_SAMPLER_HPP

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "qualla/context.hpp"
#include "qualla/detail/json.hpp"
#include "qualla/sampler.hpp"

namespace qualla {

class BasicSampler : public Sampler {
 public:
  BasicSampler(Context& ctx, const json& conf);

  virtual int32_t process(std::span<const float> logits) override;
  virtual int32_t process(std::span<const float> logits,
                          std::vector<float>& probs_out,
                          bool tok_out) override;

  virtual std::vector<int32_t> process(std::span<const float>& logits,
                                       std::vector<float>& probs,
                                       int32_t num_return) override;

  virtual bool save(const std::string& name) override;
  virtual bool restore(const std::string& name) override;
  virtual void reset() override;
  virtual void applyConfig(const qualla::json& conf) override;

  // Get sampler params
  float temp() const { return _temp; }
  size_t top_k() const { return _top_k; }
  float top_p() const { return _top_p; }

  // Set sampler params
  void temp(float t) { _temp = t; }
  void top_k(size_t k) { _top_k = k; }
  void top_p(float p) { _top_p = p; }

 protected:
  int32_t _process(std::span<const float> logits, std::vector<float>* probs_out, bool samp_tok);
  float _temp{0.1};
  size_t _top_k{0};
  float _top_p{0.8};
};

}  // namespace qualla

#endif  // QUALLA_DETAIL_BASIC_SAMPLER_HPP
