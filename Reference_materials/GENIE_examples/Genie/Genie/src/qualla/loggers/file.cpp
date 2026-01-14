//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <fmt/format.h>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>

#include "qualla/detail/config.hpp"
#include "qualla/detail/onload.hpp"
#include "qualla/detail/timer.hpp"
#include "qualla/logger.hpp"

namespace fs = std::filesystem;

namespace qualla {

class FileLogger : public Logger {
 public:
  FileLogger(const qualla::json& json) : Logger("file", json) {
    Config conf(json, "logger::file");

    fs::path out_path(conf.mandatory<std::string>("path"));

#if 0
        if (!fs::exists(out_path.parent_path()))
            throw std::runtime_error(out_path.string() + ": output directory does not exist");
#endif

    _stream.open(out_path, std::ios::out | std::ios::app);
    if (!_stream.is_open())
      throw std::runtime_error(out_path.string() + ": failed to open for writing");

    _nobuf = conf.optional<bool>("unbuf", false);
  }

  virtual ~FileLogger() { _stream.close(); }

  virtual void write(Section s, std::string_view msg) override;
  virtual void flush() override;

 private:
  std::fstream _stream;
  std::mutex _mutex;
  bool _nobuf{false};
};

void FileLogger::write(Section s, std::string_view msg) {
  qualla::Timer ts;
  uint64_t ts_sec  = ts.nsec() / 1000000000ULL;
  uint64_t ts_nsec = ts.nsec() % 1000000000ULL;

  std::lock_guard<std::mutex> guard(_mutex);
  _stream << fmt::format("{}.{:09d} QUALLA:{} {}", ts_sec, ts_nsec, this->section[s], msg)
          << std::endl;
  if (_nobuf) _stream << std::flush;
}

void FileLogger::flush() {
  std::lock_guard<std::mutex> guard(_mutex);
  _stream.flush();
}

// Registrator instance
static OnLoad regy([]() {
  Logger::__register("file", [](const json& conf) { return (Logger*)new FileLogger(conf); });
});

void needFileLogger() {}

}  // namespace qualla
