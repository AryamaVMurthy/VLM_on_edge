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

class CustomSampler : public Sampler {
 public:
  CustomSampler(Context& ctx, const json& conf);

  virtual std::vector<int32_t> process(std::span<const float>& logits,
                                       std::vector<float>& probs,
                                       int32_t num_return) override;
  virtual int32_t process(std::span<const float> logits) override;
  virtual int32_t process(std::span<const float> logits,
                          std::vector<float>& probs_out,
                          bool tok_out) override;

  virtual void applyConfig(const qualla::json& conf) override;
  virtual bool setProcessCallback(std::string name);

 protected:
  std::string _customProcessCallbackName;
  std::vector<int32_t> _process(std::span<const float>& logits, int numTokens);
};

}  // namespace qualla

#endif  // QUALLA_DETAIL_BASIC_SAMPLER_HPP
