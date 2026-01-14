//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <cstring>
#include <fstream>

#include "IOTensor.hpp"
#include "QnnApi.hpp"
#include "qnn-utils.hpp"
#include "qualla/detail/threadpool.hpp"
#include "qualla/env.hpp"

namespace qualla {

inline std::string getManagerModeStr(KVManagerMode mode) {
  if (mode == POINTER_SHIFT) return "POINTER_SHIFT";
  if (mode == SHIFT_CONCAT) return "SHIFT_CONCAT";
  if (mode == SMART_MASK) return "SMART_MASK";
  return "ERROR: KVManagerMode not found";
}

enum KVUpdateMode {
  NO_OP          = 0x0,
  CLEAR_CACHE    = 0x1,
  SET_VARIANT    = 0x2,
  UPDATE_OUTPUT  = 0x4,
  UPDATE_AND_SET = 0x8
};

inline std::string modeStr(KVUpdateMode mode) {
  if (mode == CLEAR_CACHE) return "CLEAR_CACHE";
  if (mode == SET_VARIANT) return "SET_VARIANT";
  if (mode == UPDATE_OUTPUT) return "UPDATE_OUTPUT";
  if (mode == UPDATE_AND_SET) return "UPDATE_AND_SET";
  return "NO_OP";
}

struct KVCache {
  bool is_key;
  char* buffer;
  char* output_buffer;
  int32_t n_heads;
  KVCache() {}
  KVCache(bool is_key_val, char* buffer_val, char* output_buffer_val, int32_t n_heads_val)
      : is_key(is_key_val),
        buffer(buffer_val),
        output_buffer(output_buffer_val),
        n_heads(n_heads_val) {}
};

class NewNSPKVManager {
 private:
  Env& _env;
  int _mgr_idx;  // Identify KVManager in the logs

  ThreadPool* _threadpool{nullptr};  // Threadpool for async background processing
  std::atomic_int _sync{0};

  std::vector<std::function<void()>> _update_jobs;

  KVManagerMode _mode{SMART_MASK};

  std::vector<KVCache> _kv_cache;  // <is_key, buffer, out_buffer, n_heads>
  std::vector<double> _key_scales, _value_scales;
  int32_t _max_n_heads{0};

  // Caputre states
  struct KVManagerState {
    int32_t variant;
    int32_t n_past;
    int32_t ptr_offset;
    std::vector<bool> selected;
  };

  KVManagerState _cur_state{-1, -1, 0, {}};
  KVManagerState _req_state{-1, -1, 0, {}};
  KVUpdateMode _req_mode{NO_OP};

  int32_t _counter{-1};  // Auto-increment variable for syncing updates
  int32_t n_threads{1};

  // Variant (n) stores AR-n for which the cache is currently formatted
  // The following variables are strictly dependent on variant n. Make sure to update accordingly
  size_t key_output_offset, value_output_offset;

  // Parse KV$ Tensor names here - supports past_{key,value}_{layer_idx}[_h{head_idx}]_{in,out}
  std::tuple<int, uint16_t, uint16_t> parseKVTensorName(std::string name);

  // KV Manager Utility functions
  void clearBuffer(KVCache cache) {
    std::memset(cache.buffer, _pad_value, cache.n_heads * _n_ctx * _n_embed * _bw);
  }

  bool switchKeyVariant(KVCache cache, int32_t m, int32_t n, int32_t ptr_offset);
  bool switchValueVariant(KVCache cache, int32_t m, int32_t n, int32_t ptr_offset);
  bool updateKey(KVCache cache,
                 int32_t variant,
                 int32_t n_past,
                 int32_t n_update,
                 int32_t offset,
                 const std::vector<bool>& selected);
  bool updateValue(KVCache cache,
                   int32_t variant,
                   int32_t n_past,
                   int32_t n_update,
                   int32_t offset,
                   const std::vector<bool>& selected);

  // For pointer shift
  std::map<std::string, std::pair<int, size_t>>* _alloc_info;
  bool registerPointerOffset();  // Register offsets for POINTER_SHIFT

  std::function<int32_t(int32_t)> _callback_fn;
  std::function<bool(int32_t, int32_t)> _register_pointer_fn;

 public:
  uint8_t _pad_value;  // Assumes all tensors have a common zero point @ 128
  int8_t _bw{1};       // Bitwidth of KV$ values. Defaults to 8-bit KV$
  int32_t _n_embed{-1};
  int32_t _n_ctx{-1};

  NewNSPKVManager(int idx,
                  Env& env,
                  ThreadPool* threadpool,
                  IOTensor* buffer_mgr,
                  QnnUtils::TensorMap& tensor_specs,
                  int32_t ctx_size,
                  int32_t embed_dim,
                  KVManagerMode mode);
  ~NewNSPKVManager();

  bool loadCache(std::ifstream* fs, bool is_key, int32_t n_valid, int32_t variant, int32_t n_heads);
  bool dumpCache(std::ofstream* fs, bool is_key, int32_t n_valid, int32_t n_heads);
  void updateKVCache();
  void updateKVDispatcher();
  bool updateState();
  void runKVUpdateJob(int thread_idx);  // Worker thread function
  void setTensorAllocInfo(std::map<std::string, std::pair<int, size_t>>* alloc_info) {
    _alloc_info = alloc_info;
  }
  void registerCallback(std::function<int32_t(int32_t)> callback_fn) { _callback_fn = callback_fn; }

  // TODO: Cleanup and remove this function. KVManager should handle all alloc/register for KV$
  void registerPointerOffsetFn(std::function<bool(int32_t, int32_t)> register_fn) {
    _register_pointer_fn = register_fn;
  }

  void dispatchUpdate(int32_t new_n_past, int32_t variant, const std::vector<bool>& selected);

  const size_t getNumKVTensors() const { return _kv_cache.size(); }
  const int32_t getMaxNHeads() const { return _max_n_heads; }
  int32_t getCurOffset() { return _cur_state.ptr_offset; }
  int32_t getCurVariant() { return _cur_state.variant; }
  int32_t getNPast() { return _cur_state.n_past; }
  std::vector<double>& getKeyScales() { return _key_scales; }
  std::vector<double>& getValueScales() { return _value_scales; }
};

}  // namespace qualla
