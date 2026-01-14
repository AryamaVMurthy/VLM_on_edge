//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef QUALLA_EMBEDDING_HPP
#define QUALLA_EMBEDDING_HPP

#include <atomic>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "qualla/context.hpp"
#include "qualla/detail/exports.h"
#include "qualla/detail/sentence.hpp"
#include "qualla/engine.hpp"
#include "qualla/env.hpp"
#include "qualla/tokenizer.hpp"

namespace qualla {

class Embedding : public State {
 public:
  QUALLA_API Embedding(std::shared_ptr<Env> env, const std::string& name, const qualla::json& conf);
  QUALLA_API virtual ~Embedding();

  // Encode sentence
  QUALLA_API virtual bool query(const std::string& str, std::vector<float>& output);

  // Embedding KPIs
  struct KPIs {
    struct Tps {
      size_t n_prompt;
      float prompt;
    };

    Kpi init;    // init (model load, mem allocs, etc) stats
    Kpi prompt;  // prompt processor stats
    Tps tps;     // TPS for prompt, generate, etc

    KPIs() { reset(); }
    void reset();  // reset to initial state

    QUALLA_API std::string dump(
        std::string_view sep = " ") const;  // dump KPIs as a formated string
  };

  // Get refs to various layers
  Context& context() { return *_ctx; }
  Tokenizer& tokenizer() { return *_tokenizer; }
  Engine& engine() { return *_engine; }

  // Get latest KPIs.
  // Updates TPS, etc as needed.
  QUALLA_API KPIs& kpis();

  // Get output dimensions
  QUALLA_API void output_dimensions(std::vector<std::uint32_t>& outputDimensions);

  // Create Embedding instance
  QUALLA_API static std::unique_ptr<Embedding> create(std::shared_ptr<Env> env,
                                                      const std::string& name,
                                                      const qualla::json& conf = {});
  QUALLA_API static std::unique_ptr<Embedding> create(std::shared_ptr<Env> env,
                                                      const std::string& name,
                                                      std::istream& json_stream);
  QUALLA_API static std::unique_ptr<Embedding> create(std::shared_ptr<Env> env,
                                                      const std::string& name,
                                                      const std::filesystem::path& json_path);

 protected:
  const std::string _name;
  const std::string _type;

  std::shared_ptr<Env> _env;  // Shared between multiple diaglogs and embedding
  std::unique_ptr<Context> _ctx;
  std::unique_ptr<Tokenizer> _tokenizer;
  std::unique_ptr<Engine> _engine;

  bool _input_truncation;

  std::vector<std::string> _tags;

  std::vector<std::uint32_t> _output_dimensions{};

  KPIs _kpis;
  uint32_t _n_queries{0};  // number of queries
  uint32_t _n_prompt{0};   // number of prompt tokens

  virtual bool process(std::vector<int32_t>& tokens, std::vector<float>& output);
};

}  // namespace qualla

#endif  // QUALLA_DIALOG_HPP
