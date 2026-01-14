//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once
#ifndef ACCUMULATOR_HPP
#define ACCUMULATOR_HPP
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <utility>

#include "Exception.hpp"
#include "GenieCommon.h"

namespace genie {

// Enum for supported data types
enum class QnnDataType {
    SFixed8,
    SFixed16,
    UFixed8,
    UFixed16,
    Float32,
    Unknown
};

class Accumulator {
 public:
  Accumulator(size_t bufferSize = 0);
  bool flush();
  bool append(uint8_t* data, size_t dataSizeize);
  bool append(void* src,
              std::string srcDataType,
              double srcScale,
              int32_t srcOffset,
              size_t numElements,
              uint32_t tokenNum);
  void* getData();
  size_t getDataSize();
  std::string& getDataType() { return dataType; }
  double& getScale() { return scale; }
  int32_t& getOffset() { return offset; }
  size_t getByteWidth() { return byteWidth; }

  uint32_t getTokenNum() { return embeddingTokenNum; }
  std::vector<uint32_t> getVisionParam() { return visionParam; }
  void setVisionParam(uint32_t visionPos,
                      uint32_t temporal,
                      uint32_t height,
                      uint32_t width);

  void setEncoding(std::string dType,
                   double generatorScale,
                   int32_t generatorOffset,
                   size_t generatorByteWidth);

  // Type pair struct for mapping
  struct TypePair {
    QnnDataType src;
    QnnDataType dst;
    TypePair(QnnDataType s, QnnDataType d) : src(s), dst(d) {}
    bool operator<(const TypePair& other) const {
        return std::tie(src, dst) < std::tie(other.src, other.dst);
    }
  };

  // Unified conversion function signature
  using ConvertFunc = std::function<void(void*, void*, size_t, size_t, double, int32_t, double, int32_t)>;

 private:
  void requantEmbedding(
      void* src, std::string srcDataType, double srcScale, int32_t srcOffset, size_t length);

  std::string dataType{"QNN_DATATYPE_FLOAT_32"};
  double scale{1.0};
  int32_t offset{0};
  size_t byteWidth{4};
  std::vector<uint8_t> embeddingsBuffer;
  uint32_t embeddingTokenNum{0};
  //[visionPos, temporal, height, width, ...]
  std::vector<uint32_t> visionParam;
  static const std::map<TypePair, ConvertFunc>& getConverterMap();
};
}  // namespace genie
#endif  // ACCUMULATOR_HPP