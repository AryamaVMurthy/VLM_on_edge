//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <iostream>
#include <regex>
#include <string_view>
#include <vector>

#include "qualla/detail/config.hpp"
#include "qualla/logger.hpp"

namespace qualla {

#if 0
static std::vector<std::string_view> __split(std::string_view str, char delim)
{
    std::vector<std::string_view> split;

    size_t i = 0, p = 0;

    for (; i <= str.size(); ++i) {
	if (i == str.size() || str[i] == delim) {
	    split.push_back(std::string_view(str.data() + p, i - p));
	    p = ++i;
	}
    }

    return split;
}
#endif

template <typename V>
static void __apply(uint32_t& m, std::string s, V v) {
  bool on = true;
  if (s[0] == '!') {
    on = false;
    s.erase(0, 1);
  }

  // compile regex
  std::regex re(s, std::regex::extended | std::regex::optimize);

  for (size_t i = 0; i < v.size(); ++i) {
    if (std::regex_match(std::string(v[i]), re)) {
      if (on)
        m |= (1UL << i);
      else
        m &= ~(1UL << i);
    }
  }
}

Logger::Logger(std::string_view type, const json& conf) : _type(type) {
  using strvec = std::vector<std::string>;

  if (conf.contains("mask")) {
    _runtime_mask = 0;

    if (conf["mask"].is_array()) {
      strvec v = conf["mask"].get<strvec>();
      for (auto m : v) {
        __apply(_runtime_mask, m, this->section);
      }
    } else {
      __apply(_runtime_mask, conf["mask"].get<std::string>(), this->section);
    }
  }

  if (enabled(DEBUG)) {
    std::cout << "QUALLA:DEBUG logger config : " << conf << "\n";
    std::cout << fmt::format("QUALLA:DEBUG compiled log-mask: {:#x}\n", _compiled_mask);
    std::cout << fmt::format("QUALLA:DEBUG runtime log-mask:  {:#x}\n", _runtime_mask);
  }
}

Logger::~Logger() {}

void Logger::flush() {}

// Logger registry

using Registry = std::unordered_map<std::string, Logger::Creator>;
static std::unique_ptr<Registry> registry;

void Logger::__register(const std::string& type, Creator func) {
  if (!registry) registry = std::make_unique<Registry>();

  Registry& r = *registry;
  r[type]     = func;
}

std::unique_ptr<Logger> Logger::create(const qualla::json& conf) {
  using qc         = qualla::Config;
  std::string type = qc::optional<std::string>(conf, "output", "stdout");

  if (!registry) throw std::runtime_error(type + ": logger not found");

  Registry& r = *registry;

  if (!r.contains(type)) throw std::runtime_error(type + ": logger not found");

  return std::unique_ptr<Logger>(r[type](conf));
}

std::unique_ptr<Logger> Logger::create(std::istream& json_stream) {
  return create(json::parse(json_stream));
}

std::unique_ptr<Logger> Logger::create(const std::string& json_str) {
  return create(json::parse(json_str));
}

std::vector<std::string> Logger::list() {
  std::vector<std::string> v;
  if (!registry) return v;

  Registry& r = *registry;

  for (auto k : r) v.push_back(k.first);
  return v;
}

}  // namespace qualla
