//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef QUALLA_LOGGER_HPP
#define QUALLA_LOGGER_HPP

#include <array>
#include <cstdint>
#include <string_view>
#include <vector>

#include "qualla/detail/exports.h"
#include "qualla/detail/json.hpp"
#include "qualla/detail/state.hpp"

namespace qualla {

class Logger : public State {
 public:
  enum Section {
    ERROR,
    WARN,
    INFO,
    KPIS,
    DEBUG,
    TRACE,  // Top level sections (general logging)
    ENGINE_KPIS,
    ENGINE_DEBUG,
    ENGINE_TRACE,  // Engine specific sections
    SAMPLER_KPIS,
    SAMPLER_DEBUG,
    SAMPLER_TRACE,  // Sampler specific sections
    MALLOC_DEBUG,
    KVMANAGER_TRACE  // Low-level debug sections
  };

  static constexpr std::array<std::string_view, 14> section{"ERROR",
                                                            "WARN",
                                                            "INFO",
                                                            "KPIS",
                                                            "DEBUG",
                                                            "TRACE",
                                                            "ENGINE-KPIS",
                                                            "ENGINE-DEBUG",
                                                            "ENGINE-TRACE",
                                                            "SAMPLER-KPIS",
                                                            "SAMPLER-DEBUG",
                                                            "SAMPLER-TRACE",
                                                            "MALLOC-DEBUG",
                                                            "KVMANAGER-TRACE"};

  QUALLA_API Logger(std::string_view type, const qualla::json& conf);
  QUALLA_API virtual ~Logger();

  // Check if the section is enabled
  inline bool enabled(Section s) {
    const uint32_t m = 1UL << s;
    return ((m & _compiled_mask) && (m & _runtime_mask));
  }

  // Write a record/message into the log
  QUALLA_API virtual void write(Section s, std::string_view msg) = 0;

  // Flush buffers
  QUALLA_API virtual void flush();

  // Post a record/message into the log
  // Checks if the section is enabled
  inline void post(Section s, std::string_view msg) {
    if (enabled(s)) write(s, msg);
  }

  // Post a record/message into the log using Poster function.
  // Checks if the section is enabled.
  using Poster = std::function<std::string()>;
  inline void post(Section s, Poster func) {
    if (enabled(s)) write(s, func());
  }

  // Simple helper for writing multi-line logs for specific section
  struct Helper {
    const Section sect;
    Logger& logger;

    Helper(const Section s, Logger& l) : sect(s), logger(l) {}

    void write(std::string_view msg) { logger.write(sect, msg); }
  };

  // Compose multi-line record/message into the log using Composer function.
  // Checks if the section is enabled.
  using Composer = std::function<void(Helper w)>;
  inline void compose(Section s, Composer func) {
    if (enabled(s)) func(Helper(s, *this));
  }

  // Simple wrappers for general logging
  inline void warn(std::string_view msg) { post(WARN, msg); }
  inline void warn(Poster func) { post(WARN, func); }

  inline void info(std::string_view msg) { post(INFO, msg); }
  inline void info(Poster func) { post(INFO, func); }

  inline void error(std::string_view msg) { post(ERROR, msg); }
  inline void error(Poster func) { post(ERROR, func); }

  inline void debug(std::string_view msg) { post(DEBUG, msg); }
  inline void debug(Poster func) { post(DEBUG, func); }

  // List available loggers
  QUALLA_API static std::vector<std::string> list();

  // Create Logger instance
  QUALLA_API static std::unique_ptr<Logger> create(std::istream& json_stream);
  QUALLA_API static std::unique_ptr<Logger> create(const std::string& json_str);
  QUALLA_API static std::unique_ptr<Logger> create(const qualla::json& conf = {});

  // Engine registration
  using Creator = std::function<Logger*(const qualla::json&)>;
  QUALLA_API static void __register(const std::string& type, Creator func);

 protected:
  std::string _type;

#ifdef QUALLA_COMPILED_LOGMASK
  static constexpr uint32_t _compiled_mask{uint32_t(QUALLA_COMPILED_LOGMASK)};
#else
  static constexpr uint32_t _compiled_mask{
      uint32_t(0xFFFFFFFF)};  // everything enabled at compile-time by default
#endif

  uint32_t _runtime_mask{0};
};

}  // namespace qualla

#endif  // QUALLA_LOGGER_HPP
