//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <string>

#include "IBackend.hpp"
#include "ICommandLineManager.hpp"

// This is an implementation of IBackend interface within qnn-net-run.
// NetRunBackend provides a dummy implementation of IBackend as a concrete
// implementation is needed in case there is no backend extensions library
// supplied by the user.
// This is built as part of QnnNetRun library and is used in case of no
// user supplied backend extensions implementation.
class NetRunBackend final : public IBackend {
 public:
  NetRunBackend() {}

  virtual ~NetRunBackend() {}

  virtual bool setupLogging(QnnLog_Callback_t callback, QnnLog_Level_t maxLogLevel) override {
    ignore(callback);
    ignore(maxLogLevel);
    return true;
  }

  virtual bool initialize(void* backendLibHandle) override {
    ignore(backendLibHandle);
    return true;
  }

  virtual bool setPerfProfile(PerfProfile perfProfile) override {
    ignore(perfProfile);
    return true;
  }

  virtual QnnProfile_Level_t getProfilingLevel() override { return g_profilingLevelNotSet; }

  virtual bool loadConfig(std::string configFile) override {
    ignore(configFile);
    return true;
  }

  virtual bool loadCommandLineArgs(std::shared_ptr<ICommandLineManager> clManager) override {
    ignore(clManager);
    return true;
  }

  virtual bool beforeBackendInitialize(QnnBackend_Config_t*** customConfigs,
                                       uint32_t* configCount) override {
    ignore(customConfigs);
    ignore(configCount);
    return true;
  }

  virtual bool afterBackendInitialize() override { return true; }

  virtual bool beforeContextCreate(QnnContext_Config_t*** customConfigs,
                                   uint32_t* configCount) override {
    ignore(customConfigs);
    ignore(configCount);
    return true;
  }

  virtual bool afterContextCreate() override { return true; }

  virtual bool beforeComposeGraphs(GraphConfigInfo_t*** customGraphConfigs,
                                   uint32_t* graphCount) override {
    ignore(customGraphConfigs);
    ignore(graphCount);
    return true;
  }

  virtual bool afterComposeGraphs() override { return true; }

#if QUALLA_QNN_API_VERSION >= 21700
  virtual bool beforeGraphFinalizeUpdateConfig(const char* graphName,
                                               Qnn_GraphHandle_t graphHandle,
                                               QnnGraph_Config_t*** customConfigs,
                                               uint32_t* configCount) override {
    ignore(graphName);
    ignore(graphHandle);
    ignore(customConfigs);
    ignore(configCount);
    return true;
  }
#endif

  virtual bool beforeGraphFinalize() override { return true; }

  virtual bool afterGraphFinalize() override { return true; }

  virtual bool beforeRegisterOpPackages() override { return true; }

  virtual bool afterRegisterOpPackages() override { return true; }

  virtual bool beforeExecute(const char* graphName,
                             QnnGraph_Config_t*** customConfigs,
                             uint32_t* configCount) override {
    ignore(graphName);
    ignore(customConfigs);
    ignore(configCount);
    return true;
  }

  virtual bool afterExecute() override { return true; }

  virtual bool beforeContextFree() override { return true; }

  virtual bool afterContextFree() override { return true; }

  virtual bool beforeBackendTerminate() override { return true; }

  virtual bool afterBackendTerminate() override { return true; }

  virtual bool beforeCreateFromBinary(QnnContext_Config_t*** customConfigs,
                                      uint32_t* configCount) override {
    ignore(customConfigs);
    ignore(configCount);
    return true;
  }

  virtual bool afterCreateFromBinary() override { return true; }

#if QUALLA_QNN_API_VERSION >= 21700
  virtual bool beforeCreateContextsFromBinaryList(
      std::map<std::string, std::tuple<QnnContext_Config_t**, uint32_t>>*
          contextKeyToCustomConfigsMap,
      QnnContext_Config_t*** commonCustomConfigs,
      uint32_t* commonConfigCount) override {
    ignore(contextKeyToCustomConfigsMap);
    ignore(commonCustomConfigs);
    ignore(commonConfigCount);
    return true;
  }

  virtual bool afterCreateContextsFromBinaryList() override { return true; }
#endif

  virtual bool beforeCreateDevice(QnnDevice_Config_t*** deviceConfigs,
                                  uint32_t* configCount) override {
    ignore(deviceConfigs);
    ignore(configCount);
    return true;
  }

  virtual bool afterCreateDevice() override { return true; }

  virtual bool beforeFreeDevice() override { return true; }

  virtual bool afterFreeDevice() override { return true; }

 private:
  // Utility function to ignore compiler warnings when a variable
  // is unused. Recommended by Herb Sutter in Sutter's Mill
  // instead of (void)variable.
  template <typename T>
  void ignore(const T&) {}
};
