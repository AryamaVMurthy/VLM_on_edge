//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <condition_variable>
#include <cstring>
#include <mutex>
#include <vector>

#include "nsp-graph.hpp"
#include "nsp-kvmanager.hpp"
#include "qualla/detail/threadpool.hpp"
#include "qualla/detail/timer.hpp"

namespace qualla {

struct KVState {
  int32_t counter;
  int32_t n_past;
  int32_t variant;
  std::vector<bool> selected;
  KVState() : counter(-1), n_past(-1), variant(-1) {}
  KVState(int32_t _counter, int32_t _n_past, int32_t _variant)
      : counter(_counter), n_past(_n_past), variant(_variant) {}
};

struct DispatcherState {
  DispatcherState(int split_idxVal,
                  bool activeVal,
                  bool release_lockVal,
                  QnnNspGraph* graphVal,
                  KVState currentVal,
                  KVState queuedVal,
                  KVState requestedVal)
      : split_idx(split_idxVal),
        active(activeVal),
        release_lock(release_lockVal),
        graph(graphVal),
        current(currentVal),
        queued(queuedVal),
        requested(requestedVal) {}
  int split_idx;
  bool active;        // false means inactive, i.e. no KV$ to update
  bool release_lock;  // Set to true when job is complete so we can release the lock
  QnnNspGraph* graph;
  KVState current;
  KVState queued;
  KVState requested;
};

class KVDispatcher {
 private:
  Env& _env;
  bool _threaded;
  uint64_t _cpumask{0};

  int32_t _variant{-1};

  std::vector<DispatcherState> _state;

  std::thread _dispatcher_thread;
  bool _dispatcher_terminate{false};
  bool _dispatcher_requested{false};
  bool _dispatcher_job_completed{false};
  std::mutex _dispatcher_lock;

  std::condition_variable _cv;

  // Function to add jobs to the dispatcher
  // @param split     Determines which split to update
  // @param variant   Variant of the model to use for updating
  // @param n_past    Number of past updates to include in the update
  // returns          New counter
  int32_t process(int32_t split,
                  int32_t variant,
                  int32_t n_past,
                  const std::vector<bool>& selected);

 public:
  KVDispatcher(Env& env, std::vector<QnnNspGraph>& graphs, bool threaded, uint64_t cpumask);
  ~KVDispatcher();

  // dispatch for all splits
  int32_t dispatch(int32_t variant, int32_t n_past);
  int32_t dispatch(int32_t variant, int32_t n_past, const std::vector<bool>& selected);
  int32_t dispatch(int32_t split, int32_t variant, int32_t n_past);
  int32_t dispatch(int32_t split,
                   int32_t variant,
                   int32_t n_past,
                   const std::vector<bool>& selected);

  // Callback function for worker thread to mark update job has been completed
  int32_t workerCallback(int32_t split);

  void dispatchLoop();

  void setVariant(int32_t variant) { _variant = variant; }
  int32_t getCurVariant() { return _variant; };
};
}  // namespace qualla
