//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <cmath>

#include "fmt/format.h"
#include "fmt/ranges.h"
#include "nsp-kvmanager.hpp"
#include "qualla/detail/threadpool.hpp"
#include "qualla/detail/timer.hpp"

// Copied from threadpool.cpp
#if defined(_WIN32)
#include "windows.h"

static int sched_yield(void) {
  Sleep(0);
  return 0;
}
#else
#include <sched.h>
#endif

#define __ERROR(__fmt, ...) _env.logger().post(Logger::ERROR, fmt::format(__fmt, ##__VA_ARGS__))
#define __TRACE(__fmt, ...) \
  _env.logger().post(Logger::ENGINE_TRACE, [&]() { return fmt::format(__fmt, ##__VA_ARGS__); })
#define __KVTRACE(__fmt, ...) \
  _env.logger().post(Logger::KVMANAGER_TRACE, [&]() { return fmt::format(__fmt, ##__VA_ARGS__); })

namespace qualla {

NewNSPKVManager::NewNSPKVManager(int idx,
                                 Env& env,
                                 ThreadPool* threadpool,
                                 IOTensor* buffer_mgr,
                                 QnnUtils::TensorMap& tensor_specs,
                                 int32_t ctx_size,
                                 int32_t embed_dim,
                                 KVManagerMode mode)
    : _env(env), _mgr_idx(idx), _mode(mode), _n_embed(embed_dim), _n_ctx(ctx_size) {
  // Parse KV$ Tensor names here - supports past_{key,value}_{layer_idx}[_h{head_idx}]_{in,out}
  // TODO: Enforce tensor order during allocation as well to speed up cache loops(?)
  std::map<uint32_t, QnnUtils::Tensor*> key_tensors, value_tensors;
  for (auto& [tname, tensor] : tensor_specs) {
    auto [tensor_type, layer_idx, head_idx] = parseKVTensorName(tname);
    if (tensor_type == 0) continue;
    if (tensor_type == 1)
      key_tensors[layer_idx << 16 | head_idx] = &tensor;
    else
      value_tensors[layer_idx << 16 | head_idx] = &tensor;
  }

  if (key_tensors.size() + value_tensors.size() == 0) return;

  // Calculate datatype - bitwidth and float vs quantized
  auto rt = key_tensors.size() == 0 ? value_tensors.begin()->second : key_tensors.begin()->second;
  _bw     = rt->dtype.bw();  // Assume same bitwidth for all tensors
  if (rt->dtype == QNN_DATATYPE_FLOAT_16)
    _pad_value = 0;  // For floating point inputs, pad value is 0
  else  // Currently only quantize 8-bit is supported. Will need to change to support 16-bit
    _pad_value = static_cast<uint8_t>(-rt->quantParam[0].offset);

  __TRACE("qnn-kv : {} KVManager[{} Key$ + {} Value$] : {}-bit KV$ n_embed={} n_ctx={} mode={}",
          _mgr_idx,
          key_tensors.size(),
          value_tensors.size(),
          _bw * 8,
          _n_embed,
          _n_ctx,
          getManagerModeStr(_mode));
  (void)key_output_offset;
  (void)value_output_offset;

  _kv_cache.reserve(key_tensors.size() + value_tensors.size());
  for (auto& [_, tensor] : key_tensors) {
    void* buffer = buffer_mgr->getBuffer(tensor->tensor);
    _kv_cache.emplace_back(true, (char*)buffer, (char*)buffer, tensor->dims.height);
    _key_scales.push_back(tensor->quantParam[0].scale);
  }

  for (auto& [_, tensor] : value_tensors) {
    void* buffer = buffer_mgr->getBuffer(tensor->tensor);
    _kv_cache.emplace_back(false, (char*)buffer, (char*)buffer, tensor->dims.height);
    _value_scales.push_back(tensor->quantParam[0].scale);
  }

  // Calculate _max_n_heads
  for (auto& cache : _kv_cache)
    _max_n_heads = cache.n_heads > _max_n_heads ? cache.n_heads : _max_n_heads;

  __TRACE("qnn-kv : {} KVManager[{} Key$ + {} Value$] : n_heads<={} n_embed={} n_ctx={} mode={}",
          _mgr_idx,
          key_tensors.size(),
          value_tensors.size(),
          _max_n_heads,
          _n_embed,
          _n_ctx,
          getManagerModeStr(_mode));

  if (threadpool != nullptr && threadpool->size() > 0) {
    _threadpool = threadpool;
    n_threads   = threadpool->size();
    _sync       = 0;

    _update_jobs.reserve(n_threads + 1);
    if (_mode == POINTER_SHIFT) _update_jobs.push_back([this] { this->registerPointerOffset(); });

    for (int idx = 0; idx < n_threads; idx++)
      _update_jobs.push_back([this, idx] { this->runKVUpdateJob(idx); });
  }

  _callback_fn = [](int32_t a) { return 0; };
}

NewNSPKVManager::~NewNSPKVManager() {}

// Parse KV$ Tensor names here - supports past_{key,value}_{layer_idx}[_h{head_idx}]_{in,out}
std::tuple<int, uint16_t, uint16_t> NewNSPKVManager::parseKVTensorName(std::string name) {
  if (!name.starts_with("past_")) return {0, 0, 0};

  const bool is_key = name.starts_with("past_key");
  const size_t pos0 = (is_key) ? 9 : 11;  // "past_key_" OR "past_value_"
  const size_t pos1 = name.find('_', pos0);
  const size_t pos2 = name.find('_', pos1 + 2);

  uint16_t layer_idx = 0, head_idx = 0;
  layer_idx = static_cast<uint16_t>(std::stoi(name.substr(pos0, pos1 - pos0)));
  if (pos2 != std::string::npos)
    head_idx = static_cast<uint16_t>(std::stoi(name.substr(pos1 + 2, pos2 - pos1 - 2)));

  return std::make_tuple(is_key ? 1 : 2, layer_idx, head_idx);
}

// Switch key cache from AR-m to AR-n (relative to ctx_size)
bool NewNSPKVManager::switchKeyVariant(KVCache cache, int32_t m, int32_t n, int32_t offset) {
  const size_t in_cache_dim  = (m == _n_ctx) ? _n_ctx : _n_ctx - m;
  const size_t out_cache_dim = _n_ctx - n;
  const size_t n_heads       = cache.n_heads;

  const size_t read_row_size  = in_cache_dim * _bw;
  const size_t write_row_size = out_cache_dim * _bw;
  const size_t offset_size    = offset * _bw;
  char* read_ptr              = cache.buffer;
  char* write_ptr             = cache.buffer;

  if (in_cache_dim > out_cache_dim) {
    if (_mode == POINTER_SHIFT || _mode == SHIFT_CONCAT) {  // Left padded KV$
      read_ptr += read_row_size - write_row_size + offset_size;
      write_ptr += offset_size;
    }

    for (int i = 0; i < n_heads * _n_embed; i++) {
      std::memmove(write_ptr, read_ptr, write_row_size);
      read_ptr += read_row_size;
      write_ptr += write_row_size;
    }
  } else {
    const size_t block_size_delta = write_row_size - read_row_size;
    read_ptr += (n_heads * _n_embed - 1) * read_row_size + offset_size;
    write_ptr += (n_heads * _n_embed - 1) * write_row_size + offset_size;
    char* pad_ptr = (_mode == SMART_MASK) ? write_ptr + read_row_size : write_ptr;
    if (_mode == POINTER_SHIFT || _mode == SHIFT_CONCAT) write_ptr += block_size_delta;

    for (int i = 0; i < n_heads * _n_embed; i++) {
      std::memmove(write_ptr, read_ptr, read_row_size);
      std::memset(pad_ptr, _pad_value, block_size_delta);
      read_ptr -= read_row_size;
      write_ptr -= write_row_size;
      pad_ptr -= write_row_size;
    }
  }

  return true;
}

// Switch value cache from AR-m to AR-n (relative to ctx_size)
bool NewNSPKVManager::switchValueVariant(KVCache cache, int32_t m, int32_t n, int32_t offset) {
  const size_t in_cache_dim  = (m == _n_ctx) ? _n_ctx : _n_ctx - m;
  const size_t out_cache_dim = _n_ctx - n;
  const size_t n_heads       = cache.n_heads;

  const size_t read_block_size  = in_cache_dim * _n_embed * _bw;
  const size_t write_block_size = out_cache_dim * _n_embed * _bw;
  const size_t offset_size      = offset * _n_embed * _bw;
  char* read_ptr                = cache.buffer;
  char* write_ptr               = cache.buffer;

  if (in_cache_dim > out_cache_dim) {
    if (_mode == POINTER_SHIFT || _mode == SHIFT_CONCAT) {  // Left padded KV$
      read_ptr += read_block_size - write_block_size + offset_size;
      write_ptr += offset_size;
    }

    for (int i = 0; i < n_heads; i++) {
      std::memmove(write_ptr, read_ptr, write_block_size);
      read_ptr += read_block_size;
      write_ptr += write_block_size;
    }
  } else {
    const size_t block_size_delta = write_block_size - read_block_size;
    read_ptr += (n_heads - 1) * read_block_size + offset_size;
    write_ptr += (n_heads - 1) * write_block_size + offset_size;
    char* pad_ptr = (_mode == SMART_MASK) ? write_ptr + read_block_size : write_ptr;
    if (_mode == POINTER_SHIFT || _mode == SHIFT_CONCAT) write_ptr += block_size_delta;

    for (int i = 0; i < n_heads; i++) {
      std::memmove(write_ptr, read_ptr, read_block_size);
      std::memset(pad_ptr, _pad_value, block_size_delta);
      read_ptr -= read_block_size;
      write_ptr -= write_block_size;
      pad_ptr -= write_block_size;
    }
  }

  return true;
}

bool NewNSPKVManager::updateKey(KVCache cache,
                                int32_t variant,
                                int32_t n_past,
                                int32_t n_update,
                                int32_t offset,
                                const std::vector<bool>& selected) {
  char* dst = cache.buffer;
  char* src = cache.output_buffer;

  const int32_t n_iter      = cache.n_heads * _n_embed;
  const int32_t iter_size   = (_n_ctx - variant) * _bw;
  const int32_t copy_size   = abs(n_update) * _bw;
  const int32_t offset_size = offset * _bw;
  const int32_t past_size   = n_past * _bw;
  const int32_t out_size    = variant * _bw;

  // Shrink KV$ by removing n_update entries
  if (n_update < 0) {
    if (_mode == SHIFT_CONCAT) {
      std::memmove(dst + copy_size, dst, n_iter * iter_size - copy_size);
      std::memset(dst, _pad_value, copy_size);
    } else {
      char* write_ptr = dst;
      if (_mode == POINTER_SHIFT) write_ptr += offset_size + iter_size - copy_size;
      if (_mode == SMART_MASK) write_ptr += past_size - copy_size;

      for (int32_t i = 0; i < n_iter; i++) {
        std::memset(write_ptr, _pad_value, copy_size);
        write_ptr += iter_size;
      }
    }
    return true;
  }

  // Concatenate output into the KV$ buffers
  if (_mode == SHIFT_CONCAT)  // Shift KV$ buffer if necessary
    std::memmove(dst, dst + copy_size, n_iter * iter_size - copy_size);

  char* read_ptr  = src;  // output_buffer
  char* write_ptr = dst;  // input_buffer

  if (_mode == POINTER_SHIFT) write_ptr += offset_size + iter_size;
  if (_mode == SHIFT_CONCAT) write_ptr += iter_size - copy_size;
  if (_mode == SMART_MASK) write_ptr += past_size;

  if (selected.empty()) {
    for (int32_t i = 0; i < n_iter; i++) {
      std::memcpy(write_ptr, read_ptr, copy_size);
      write_ptr += iter_size;
      read_ptr += out_size;
    }
  } else {
    for (int32_t i = 0; i < n_iter; i++) {
      auto wp = write_ptr, rp = read_ptr;
      for (auto sel : selected) {
        if (sel)
          for (int i = 0; i < _bw; i++) *wp++ = rp[i];
        rp += _bw;
      }
      write_ptr += iter_size;
      read_ptr += out_size;
    }
  }
  return true;
}

bool NewNSPKVManager::updateValue(KVCache cache,
                                  int32_t variant,
                                  int32_t n_past,
                                  int32_t n_update,
                                  int32_t offset,
                                  const std::vector<bool>& selected) {
  char* dst = cache.buffer;
  char* src = cache.output_buffer;

  const int32_t n_iter      = cache.n_heads;
  const int32_t iter_size   = (_n_ctx - variant) * _n_embed * _bw;
  const int32_t copy_size   = abs(n_update) * _n_embed * _bw;
  const int32_t offset_size = offset * _n_embed * _bw;
  const int32_t past_size   = n_past * _n_embed * _bw;
  const int32_t out_size    = variant * _n_embed * _bw;

  if (n_update < 0) {
    if (_mode == SHIFT_CONCAT) {
      std::memmove(dst + copy_size, dst, n_iter * iter_size - copy_size);
      std::memset(dst, _pad_value, copy_size);
    } else {
      char* write_ptr = dst;
      if (_mode == POINTER_SHIFT) write_ptr += offset_size + iter_size - copy_size;
      if (_mode == SMART_MASK) write_ptr += past_size - copy_size;

      for (int32_t i = 0; i < n_iter; i++) {
        std::memset(write_ptr, _pad_value, copy_size);
        write_ptr += iter_size;
      }
    }
    return true;
  }

  if (_mode == SHIFT_CONCAT)  // Shift KV$ buffer if necessary
    std::memmove(dst, dst + copy_size, n_iter * iter_size - copy_size);

  // Concatenate output into the KV$ buffers
  char* read_ptr  = src;
  char* write_ptr = dst;

  if (_mode == POINTER_SHIFT) write_ptr += offset_size + iter_size;
  if (_mode == SHIFT_CONCAT) write_ptr += iter_size - copy_size;
  if (_mode == SMART_MASK) write_ptr += past_size;

  if (selected.empty()) {
    for (int i = 0; i < n_iter; i++) {
      std::memcpy(write_ptr, read_ptr, copy_size);
      write_ptr += iter_size;
      read_ptr += out_size;
    }
  } else {
    for (int i = 0; i < n_iter; i++) {
      auto wp = write_ptr, rp = read_ptr;
      for (auto sel : selected) {
        if (sel) {
          std::memcpy(wp, rp, _n_embed * _bw);
          wp += _n_embed * _bw;
        }
        rp += _n_embed * _bw;
      }
      write_ptr += iter_size;
      read_ptr += out_size;
    }
  }
  return true;
}

bool NewNSPKVManager::registerPointerOffset() {
  int32_t variant    = _req_state.variant;
  int32_t ptr_offset = _req_state.ptr_offset;
  __KVTRACE("qnn-kv : graph[{}] pointerShift({} @ AR-{})", _mgr_idx, ptr_offset, variant);
  _register_pointer_fn(variant, ptr_offset * _bw);

  if (_threadpool != nullptr) {
    const int rem = --_sync;
    __KVTRACE("qnn-kv : graph[{}] pointerShift complete ({} remain)", _mgr_idx, rem);
    if (rem == 0) updateState();
  }
  return true;
}

void NewNSPKVManager::updateKVCache() {
  __TRACE("qnn-kv : graph[{}] updateState to AR-{}(n_past={}, ptr={})",
          _mgr_idx,
          _req_state.variant,
          _req_state.n_past,
          _req_state.ptr_offset);

  if (_cur_state.variant != _req_state.variant) {
    for (KVCache& cache : _kv_cache) {
      const int32_t dim_size = _n_ctx - _req_state.variant;
      cache.output_buffer    = cache.buffer + dim_size * cache.n_heads * _n_embed * _bw;

      if (_mode == POINTER_SHIFT)
        cache.output_buffer += cache.is_key ? _n_ctx * _bw : _n_ctx * _n_embed * _bw;
    }
  }
  _cur_state = _req_state;
}
void NewNSPKVManager::updateKVDispatcher() { _counter = _callback_fn(_mgr_idx); }

bool NewNSPKVManager::updateState() {
  updateKVCache();
  updateKVDispatcher();
  return true;
}

// Function executes on the threadpool - called once per thread.
// Assumes the lock is properly attained by this point
void NewNSPKVManager::runKVUpdateJob(int thread_idx) {
  __KVTRACE("qnn-kv : graph[{}] tid[{}] kv-update started. {} ",
            _mgr_idx,
            thread_idx,
            modeStr(_req_mode));
  (void)&modeStr;
  int job_count = 1 + ((getNumKVTensors() - 1) / n_threads);  // Number of jobs per thread
  int end_idx   = job_count * (thread_idx + 1);
  if (end_idx > getNumKVTensors()) end_idx = getNumKVTensors();

  for (int idx = job_count * thread_idx; idx < end_idx; idx++) {
    KVCache& cache = _kv_cache[idx];

    auto& [variant, n_past, ptr_offset, selected] = _cur_state;
    const int32_t n_update                        = _req_state.n_past - n_past;

    if (cache.is_key) {
      if (_req_mode == CLEAR_CACHE) clearBuffer(cache);
      if (_req_mode == UPDATE_OUTPUT || _req_mode == UPDATE_AND_SET) {
        updateKey(cache, variant, n_past, n_update, ptr_offset, _req_state.selected);
      }
      if (_req_mode == SET_VARIANT || _req_mode == UPDATE_AND_SET) {
        switchKeyVariant(cache, variant, _req_state.variant, _req_state.ptr_offset);
      }
    } else {
      if (_req_mode == CLEAR_CACHE) clearBuffer(cache);
      if (_req_mode == UPDATE_OUTPUT || _req_mode == UPDATE_AND_SET) {
        updateValue(cache, variant, n_past, n_update, ptr_offset, _req_state.selected);
      }
      if (_req_mode == SET_VARIANT || _req_mode == UPDATE_AND_SET) {
        switchValueVariant(cache, variant, _req_state.variant, _req_state.ptr_offset);
      }
    }
  }

  if (_threadpool != nullptr) {
    const int rem = --_sync;
    __KVTRACE("qnn-kv : graph[{}] tid[{}] kv-update ({} remain)", _mgr_idx, thread_idx, rem);
    if (rem == 0) updateState();
  } else  // Without threading, this is only called once so we can updateState() immediately
    updateState();
}

void NewNSPKVManager::dispatchUpdate(int32_t n_past,
                                     int32_t variant,
                                     const std::vector<bool>& selected) {
  __KVTRACE("qnn-kv : graph[{}] dispatchUpdate AR-{}(n_past={}, ptr={}) -> AR-{}(n_past={})",
            _mgr_idx,
            _cur_state.variant,
            _cur_state.n_past,
            _cur_state.ptr_offset,
            variant,
            n_past);

  _req_state = {variant, n_past, _cur_state.ptr_offset, selected};

  if (_req_state.n_past == 0) {
    _req_mode             = CLEAR_CACHE;
    _req_state.ptr_offset = 0;

    // Nothing to be done iff
    // - Requested variant is BERT Mode, i.e. takes no input (new_variant == _n_ctx)
    // - Cache is already empty (n_past == 0)
    if (_req_state.variant == _n_ctx || _cur_state.n_past == 0) _req_mode = NO_OP;
  } else if (_req_state.n_past == _cur_state.n_past) {
    _req_mode = SET_VARIANT;
    // Nothing needs to be done iff
    // - Cache is empty (n_past == 0). Might want to check for BERT->AR-1
    // - Requested variant is already set (new_variant == cur_variant)
    // - Requested variant is BERT Mode, i.e. takes no input (new_variant == _n_ctx)
    if (_cur_state.n_past == 0 || _req_state.variant == _n_ctx ||
        _req_state.variant == _cur_state.variant)
      _req_mode = NO_OP;
    if (_req_state.variant == _n_ctx) _req_state.ptr_offset = 0;

  } else if (_req_state.n_past < _cur_state.n_past) {
    _req_mode = UPDATE_OUTPUT;
    if (_mode == POINTER_SHIFT) _req_state.ptr_offset -= (_cur_state.n_past - _req_state.n_past);

  } else if (_req_state.variant == _cur_state.variant) {  // UPDATE_OUTPUT
    _req_mode = UPDATE_OUTPUT;
    if (_cur_state.variant == _n_ctx)
      _req_mode = NO_OP;
    else if (_mode == POINTER_SHIFT)
      _req_state.ptr_offset += (_req_state.n_past - _cur_state.n_past);

  } else {
    _req_mode = UPDATE_AND_SET;

    if (_cur_state.variant == _n_ctx)
      _req_mode = SET_VARIANT;
    else if (_req_state.variant == _n_ctx) {
      _req_state.n_past = 0;
      _req_mode         = NO_OP;  // If we're switching to BERT-Mode, nothing to do
    }

    if (_req_mode == UPDATE_AND_SET && _cur_state.variant != _n_ctx && _mode == POINTER_SHIFT)
      _req_state.ptr_offset += (_req_state.n_past - _cur_state.n_past);
  }

  __KVTRACE("qnn-kv : graph[{}] Processing {} AR-{}(n_past={}, ptr={})",
            _mgr_idx,
            modeStr(_req_mode),
            _req_state.variant,
            _req_state.n_past,
            _req_state.ptr_offset);

  if (_req_mode == NO_OP) {
    // TODO: Think about this case a bit more. Any other cases we want to registerPtrOffset()?
    bool needs_register_ptr =
        (_mode == POINTER_SHIFT && (_cur_state.variant != _req_state.variant ||
                                    _cur_state.ptr_offset != _req_state.ptr_offset));

    if (needs_register_ptr) {
      if (_threadpool != nullptr) {
        _sync += 1;
        registerPointerOffset();
      } else {
        registerPointerOffset();
        updateState();
      }
    } else
      updateState();
    return;
  }

  if (_threadpool != nullptr) {
    _sync += _update_jobs.size();
    _threadpool->enqueue(_update_jobs);
  } else {
    runKVUpdateJob(0);
    if (_mode == POINTER_SHIFT) registerPointerOffset();
    updateState();
  }
}

bool NewNSPKVManager::loadCache(
    std::ifstream* fs, bool is_key, int32_t n_valid, int32_t variant, int32_t n_heads) {
  __TRACE("qnn-kv : KVManager[{}] load cache", _mgr_idx);
  const size_t cache_dim = (variant == _n_ctx) ? _n_ctx : _n_ctx - variant;
  const size_t iter_size = (is_key) ? cache_dim * _bw : cache_dim * _n_embed * _bw;
  const size_t copy_size = (is_key) ? n_valid * _bw : n_valid * _n_embed * _bw;

  for (KVCache& cache : _kv_cache) {
    if (cache.is_key != is_key) continue;

    clearBuffer(cache);
    const int n_iter = (is_key) ? cache.n_heads * _n_embed : cache.n_heads;
    char* data       = (char*)cache.buffer;
    if (_mode == POINTER_SHIFT || _mode == SHIFT_CONCAT) data += iter_size - copy_size;
    for (int i = 0; i < n_iter; i++) {
      fs->read(data, copy_size);
      data += iter_size;  // Jump to the next row/block (depending on type)
    }

    if (n_heads > cache.n_heads)
      fs->seekg((n_heads - cache.n_heads) * _n_embed * n_valid * _bw, std::ios::cur);
  }

  _req_state = {variant, n_valid, 0};
  updateKVCache();

  return true;
}

bool NewNSPKVManager::dumpCache(std::ofstream* fs, bool is_key, int32_t n_valid, int32_t n_heads) {
  __TRACE("qnn-kv : graph[{}] dump cache", _mgr_idx);
  const int32_t variant    = _cur_state.variant;
  const int32_t ptr_offset = _cur_state.ptr_offset;
  const size_t cache_dim   = (variant == _n_ctx) ? _n_ctx : _n_ctx - variant;

  const size_t iter_size   = (is_key) ? cache_dim * _bw : cache_dim * _n_embed * _bw;
  const size_t copy_size   = (is_key) ? n_valid * _bw : n_valid * _n_embed * _bw;
  const size_t offset_size = (is_key) ? ptr_offset * _bw : ptr_offset * _n_embed * _bw;

  for (KVCache& cache : _kv_cache) {
    if (cache.is_key != is_key) continue;

    const int n_iter = (is_key) ? cache.n_heads * _n_embed : cache.n_heads;
    char* data       = (char*)cache.buffer;
    if (_mode == POINTER_SHIFT || _mode == SHIFT_CONCAT)
      data += offset_size + iter_size - copy_size;
    for (int i = 0; i < n_iter; i++) {
      fs->write(data, copy_size);
      data += iter_size;  // Jump to the next row/block (depending on type)
    }

    if (n_heads > cache.n_heads)
      fs->seekp((n_heads - cache.n_heads) * _n_embed * n_valid * _bw, std::ios::cur);
  }
  return true;
}
}  // namespace qualla
