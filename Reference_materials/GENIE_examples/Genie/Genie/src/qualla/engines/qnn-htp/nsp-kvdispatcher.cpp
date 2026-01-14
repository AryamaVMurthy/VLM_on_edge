//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include "fmt/format.h"
#include "fmt/ranges.h"
#include "nsp-kvdispatcher.hpp"

#define __ERROR(__fmt, ...) _env.logger().post(Logger::ERROR, fmt::format(__fmt, ##__VA_ARGS__))
#define __KVTRACE(__fmt, ...) \
  _env.logger().post(Logger::KVMANAGER_TRACE, [&]() { return fmt::format(__fmt, ##__VA_ARGS__); })

// Copied from threadpool.cpp
#if defined(_WIN32)
#define NOGDI
#include "windows.h"

static bool __thread_affinity(uint64_t mask) {
  HANDLE h    = GetCurrentThread();
  DWORD_PTR m = mask;

  m = SetThreadAffinityMask(h, m);

  return m != 0;
}

static int sched_yield(void) {
  Sleep(0);
  return 0;
}

#elif defined(__APPLE__)
static bool __thread_affinity(uint64_t mask) { return true; }

#else  // posix?
#include <errno.h>
#include <sched.h>
#include <string.h>

static bool __thread_affinity(uint64_t mask) {
  cpu_set_t cpuset;
  int32_t err;

  CPU_ZERO(&cpuset);

  for (uint32_t i = 0; i < 64; i++) {
    if ((1ULL << i) & mask) {
      CPU_SET(i, &cpuset);
    }
  }

#ifdef __ANDROID__
  err = sched_setaffinity(0, sizeof(cpuset), &cpuset);
  if (err < 0) {
    err = errno;
  }
#else
  err = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
#endif
  if (err != 0) {
    fprintf(stderr,
            "warn: failed to set affinity mask 0x%llx (err %d: %s)\n",
            (unsigned long long)mask,
            err,
            strerror(err));
    return false;
  }

  return true;
}

#endif

#ifdef _MSC_VER

static inline void __cpu_relax(void) { YieldProcessor(); }

#else

#if defined(__aarch64__)

static inline void __cpu_relax(void) { __asm__ volatile("yield" ::: "memory"); }

#else

static inline void __cpu_relax(void) { __asm__ volatile("rep; nop" ::: "memory"); }

#endif
#endif

namespace qualla {

KVDispatcher::KVDispatcher(Env& env,
                           std::vector<QnnNspGraph>& graphs,
                           bool threaded,
                           uint64_t cpumask)
    : _env(env), _threaded(threaded), _cpumask(cpumask) {
  (void)_cpumask;

  int32_t idx = 0;
  for (QnnNspGraph& graph : graphs) {
    if (_threaded)
      graph.kvmanager->registerCallback(
          [this](int32_t split) { return this->workerCallback(split); });

    // Initialize new DispatcherState()
    bool active = (graph.kvmanager->getNumKVTensors() > 0);
    _state.emplace_back(idx, active, false, &graph, KVState(), KVState(), KVState());
    idx++;
  }

  if (_threaded) _dispatcher_thread = std::thread(&KVDispatcher::dispatchLoop, this);
}

KVDispatcher::~KVDispatcher() {
  if (_threaded) {
    _dispatcher_terminate = true;
    _cv.notify_all();
    _dispatcher_thread.join();
  }
}

int32_t KVDispatcher::process(int32_t split,
                              int32_t variant,
                              int32_t n_past,
                              const std::vector<bool>& selected) {
  DispatcherState& state = _state[split];

  state.requested.n_past   = n_past;
  state.requested.variant  = variant;
  state.requested.selected = selected;
  return ++state.requested.counter;
}

int32_t KVDispatcher::dispatch(int32_t split, int32_t variant, int32_t n_past) {
  return dispatch(split, variant, n_past, {});
}
int32_t KVDispatcher::dispatch(int32_t split,
                               int32_t variant,
                               int32_t n_past,
                               const std::vector<bool>& selected) {
  _variant = variant;

  if (!_threaded) {
    if (_state[split].active)
      _state[split].graph->kvmanager->dispatchUpdate(n_past, variant, selected);
    return 0;
  }

  if (!_state[split].active)  // Increment current counter and return new value
    return _state[split].current.counter = process(split, variant, n_past, selected);

  int32_t updated_idx;
  {
    std::lock_guard lk(_dispatcher_lock);
    updated_idx           = process(split, variant, n_past, selected);
    _dispatcher_requested = true;
  }

  _cv.notify_one();
  return updated_idx;
}

int32_t KVDispatcher::dispatch(int32_t variant, int32_t n_past) {
  return dispatch(variant, n_past, std::vector<bool>{});
}

int32_t KVDispatcher::dispatch(int32_t variant, int32_t n_past, const std::vector<bool>& selected) {
  _variant = variant;

  if (!_threaded) {
    for (auto& s : _state)
      if (s.active) s.graph->kvmanager->dispatchUpdate(n_past, variant, selected);
    return 0;
  }

  int32_t global_updated_idx = -1;
  {
    std::lock_guard lk(_dispatcher_lock);

    for (auto& s : _state) {
      if (!s.active) {
        global_updated_idx = (s.current.counter = process(s.split_idx, variant, n_past, selected));
        continue;
      }

      int32_t updated_idx = process(s.split_idx, variant, n_past, selected);
      if (global_updated_idx == -1)
        global_updated_idx = updated_idx;
      else if (global_updated_idx != updated_idx) {
        // Something went wrong. States are not in sync
        __ERROR(
            "qnn-kv: Dispatcher states out of sync - {} vs {}", global_updated_idx, updated_idx);
      }
    }
    _dispatcher_requested = true;
  }

  _cv.notify_one();
  return global_updated_idx;
}

void KVDispatcher::dispatchLoop() {
  // if (_cpumask) __thread_affinity(_cpumask);
  (void)&__thread_affinity;
  (void)&__cpu_relax;

  // loop dispatch
  std::vector<int32_t> dispatch_queue;
  dispatch_queue.reserve(_state.size());
  std::unique_lock lk(_dispatcher_lock, std::defer_lock);

  while (true) {
    lk.lock();
    _cv.wait(lk, [this] {
      return _dispatcher_terminate || _dispatcher_requested || _dispatcher_job_completed;
    });

    // On exit, release all locks
    if (_dispatcher_terminate) {
      for (auto& s : _state) {
        if (s.active && (s.release_lock || s.current.counter != s.queued.counter))
          s.graph->releaseLock("dispatcher_terminate");
      }
      lk.unlock();
      break;
    }

    __KVTRACE("qnn-kv: Dispatcher ({}, {})", _dispatcher_requested, _dispatcher_job_completed);

    // When a job is complete, release all relevant locks
    if (_dispatcher_job_completed) {
      for (auto& s : _state) {
        if (s.release_lock) {
          s.graph->releaseLock("kv-update");
          s.release_lock = false;
        }
      }
    }

    for (auto& s : _state) {
      if (!s.active) {
        s.current = s.requested;
        continue;
      }

      auto& current   = s.current;
      auto& queued    = s.queued;
      auto& requested = s.requested;

      // There is no new work to be done, OR
      // KVManager is already working on a job on this split. Wait for completion.
      if (queued.counter == requested.counter || current.counter != queued.counter) continue;

      // Requested change has already been completed
      if (current.n_past == requested.n_past && current.variant == requested.variant) {
        s.graph->_counter = current.counter = queued.counter = requested.counter;
        s.graph->wakeUpLock();
        continue;
      }

      // Job has been requested but not yet dispatched
      s.queued = s.requested;
      dispatch_queue.emplace_back(s.split_idx);
    }

    _dispatcher_job_completed = false;  // Be ready for next job completion
    _dispatcher_requested     = false;  // Be ready for next job request

    lk.unlock();

    // Dispatch jobs
    for (auto split : dispatch_queue) {
      DispatcherState& s = _state[split];
      s.graph->waitForLock("kv-update");
      s.graph->kvmanager->dispatchUpdate(s.queued.n_past, s.queued.variant, s.queued.selected);
    }
    dispatch_queue.clear();
  }
  __KVTRACE("qnn-kv : Dispatcher terminating");
}

int32_t KVDispatcher::workerCallback(int32_t split) {
  __KVTRACE("qnn-kv : graph[{}] workerCallback()", split);
  {
    std::lock_guard lk(_dispatcher_lock);
    // Update relevant job counters
    _state[split].current         = _state[split].queued;
    _state[split].graph->_counter = _state[split].current.counter;
    _state[split].release_lock    = true;
    _dispatcher_job_completed     = true;
  }

  _cv.notify_one();
  return _state[split].current.counter;
}

}  // namespace qualla
