//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef QUALLA_DETAIL_KV_SHARE_DIALOG_HPP
#define QUALLA_DETAIL_KV_SHARE_DIALOG_HPP

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>
#include <filesystem>

#include "qualla/detail/json.hpp"
#include "qualla/dialog.hpp"
#include "qualla/env.hpp"

namespace fs = std::filesystem;

namespace qualla {

  class KvShareDialog : public Dialog {
  public:
    KvShareDialog(std::shared_ptr<Env> env, const std::string& name, const json& conf)
      : Dialog(env, name, conf) {}

    virtual bool process(std::vector<int32_t>& tokens, Dialog::Callback callback) override;

    virtual bool process(std::vector<int32_t>& tokens, qualla::DialogCallback callback) override;

    virtual void reset() override;

    bool convertKV(const fs::path& cache_dir);
  };

}  // namespace qualla

#endif  // QUALLA_DETAIL_KV_SHARE_DIALOG_HPP
