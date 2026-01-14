//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <fmt/format.h>

#include <iostream>

#include "qualla/detail/config.hpp"
#include "qualla/detail/onload.hpp"
#include "qualla/logger.hpp"

namespace qualla {

class StdoutLogger : public Logger {
 public:
  StdoutLogger(const json& conf) : Logger("stdout", conf) {
    using qc = qualla::Config;
    _unbuf   = qc::optional<bool>(conf, "unbuf", false);
  }

  virtual void write(Section s, std::string_view msg) override;
  virtual void flush() override;

 private:
  bool _unbuf{false};
};

void StdoutLogger::write(Section s, std::string_view msg) {
  std::cout << fmt::format("QUALLA:{} {}\n", this->section[s], msg);
  if (_unbuf) std::cout << std::flush;
}

void StdoutLogger::flush() { std::cout << std::flush; }

// Registrator instance
static OnLoad regy([]() {
  Logger::__register("stdout", [](const json& conf) { return (Logger*)new StdoutLogger(conf); });
});

void needStdoutLogger() {}

}  // namespace qualla
