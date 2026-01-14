//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>

#include "Accumulator.hpp"
#include "QnnTypes.h"

using namespace genie;

Accumulator::Accumulator(size_t bufferSize) { embeddingsBuffer.reserve(bufferSize); }

bool Accumulator::append(uint8_t* data, size_t dataSize) {
  embeddingsBuffer.insert(embeddingsBuffer.end(), data, data + dataSize);
  return true;
}

bool Accumulator::append(void* src,
                         std::string srcDataType,
                         double srcScale,
                         int32_t srcOffset,
                         size_t numElements,
                         uint32_t tokenNum) {
  requantEmbedding(src, srcDataType, srcScale, srcOffset, numElements);
  embeddingTokenNum += tokenNum;
  return true;
}

bool Accumulator::flush() {
  embeddingsBuffer.clear();
  embeddingTokenNum = 0;
  visionParam.clear();
  return true;
}

void* Accumulator::getData() { return embeddingsBuffer.data(); }

size_t Accumulator::getDataSize() { return embeddingsBuffer.size(); }

void Accumulator::setEncoding(std::string dType,
                              double generatorScale,
                              int32_t generatorOffset,
                              size_t generatorByteWidth) {
  byteWidth = generatorByteWidth;
  offset    = generatorOffset;
  scale     = generatorScale;
  dataType  = dType;
}

// Convert string to enum
static QnnDataType toQnnDataType(const std::string& typeStr) {
  if (typeStr == "QNN_DATATYPE_SFIXED_POINT_8")   return QnnDataType::SFixed8;
  if (typeStr == "QNN_DATATYPE_SFIXED_POINT_16")  return QnnDataType::SFixed16;
  if (typeStr == "QNN_DATATYPE_UFIXED_POINT_8")   return QnnDataType::UFixed8;
  if (typeStr == "QNN_DATATYPE_UFIXED_POINT_16")  return QnnDataType::UFixed16;
  if (typeStr == "QNN_DATATYPE_FLOAT_32")         return QnnDataType::Float32;
  return QnnDataType::Unknown;
}

// fixed-point to fixed-point
template <typename SrcT, typename DstT>
struct RequantConvert {
  static void run(void* src, void* dst,
                  size_t dstStartAddress, size_t length,
                  double requantScale, int32_t requantOffset,
                  double /*srcScale*/, int32_t /*srcOffset*/) {
    SrcT* s = static_cast<SrcT*>(src);
    DstT* d = static_cast<DstT*>(dst);
    for (size_t i = 0; i < length; ++i) {
      const double val = static_cast<double>(s[i]) * requantScale + requantOffset;
      d[dstStartAddress + i] = static_cast<DstT>(std::round(val));
    }
  }
};

// fixed-point to float
template <typename SrcT>
struct RequantConvert<SrcT, float> {
  static void run(void* src, void* dst,
                  size_t dstStartAddress, size_t length,
                  double /*requantScale*/, int32_t /*requantOffset*/,
                  double srcScale, int32_t srcOffset) {
    SrcT* s = static_cast<SrcT*>(src);
    float* d = static_cast<float*>(dst);
    for (size_t i = 0; i < length; ++i) {
      // d = (s + srcOffset) * srcScale
      d[dstStartAddress + i] =
          static_cast<float>((static_cast<double>(s[i]) + srcOffset) * srcScale);
    }
  }
};

// float to fixed-point
template <typename DstT>
struct RequantConvert<float, DstT> {
  static void run(void* src, void* dst,
                  size_t dstStartAddress, size_t length,
                  double scale, int32_t offset,
                  double /*srcScale*/, int32_t /*srcOffset*/) {
    float* s = static_cast<float*>(src);
    DstT* d = static_cast<DstT*>(dst);
    for (size_t i = 0; i < length; ++i) {
      // val = s/scale - offset
      const double val = static_cast<double>(s[i]) / scale - offset;
      d[dstStartAddress + i] = static_cast<DstT>(std::round(val));
    }
  }
};

// float to float
template <>
struct RequantConvert<float, float> {
  static void run(void* src, void* dst,
                  size_t dstStartAddress, size_t length,
                  double /*scale*/, int32_t /*offset*/,
                  double /*srcScale*/, int32_t /*srcOffset*/) {
    std::memcpy(static_cast<float*>(dst) + dstStartAddress,
                src, length * sizeof(float));
  }
};

// Unified entry point: generic function template
template <typename SrcT, typename DstT>
void requantConvertFunc(void* src, void* dst,
                        size_t dstStartAddress, size_t length,
                        double requantScale, int32_t requantOffset,
                        double srcScale, int32_t srcOffset) {
  RequantConvert<SrcT, DstT>::run(
      src, dst, dstStartAddress, length,
      requantScale, requantOffset, srcScale, srcOffset);
}

// Static mapping table initialization
const std::map<Accumulator::TypePair, Accumulator::ConvertFunc>& Accumulator::getConverterMap() {
  static const std::map<TypePair, ConvertFunc> m = {
    // SFixed8
    { TypePair(QnnDataType::SFixed8,  QnnDataType::SFixed8),   &requantConvertFunc<int8_t,   int8_t>   },
    { TypePair(QnnDataType::SFixed8,  QnnDataType::SFixed16),  &requantConvertFunc<int8_t,   int16_t>  },
    { TypePair(QnnDataType::SFixed8,  QnnDataType::Float32),   &requantConvertFunc<int8_t,   float>    },
    // SFixed16
    { TypePair(QnnDataType::SFixed16, QnnDataType::SFixed8),   &requantConvertFunc<int16_t,  int8_t>   },
    { TypePair(QnnDataType::SFixed16, QnnDataType::SFixed16),  &requantConvertFunc<int16_t,  int16_t>  },
    { TypePair(QnnDataType::SFixed16, QnnDataType::Float32),   &requantConvertFunc<int16_t,  float>    },
    // UFixed8
    { TypePair(QnnDataType::UFixed8,  QnnDataType::UFixed8),   &requantConvertFunc<uint8_t,  uint8_t>  },
    { TypePair(QnnDataType::UFixed8,  QnnDataType::UFixed16),  &requantConvertFunc<uint8_t,  uint16_t> },
    { TypePair(QnnDataType::UFixed8,  QnnDataType::Float32),   &requantConvertFunc<uint8_t,  float>    },
    // UFixed16
    { TypePair(QnnDataType::UFixed16, QnnDataType::UFixed8),   &requantConvertFunc<uint16_t, uint8_t>  },
    { TypePair(QnnDataType::UFixed16, QnnDataType::UFixed16),  &requantConvertFunc<uint16_t, uint16_t> },
    { TypePair(QnnDataType::UFixed16, QnnDataType::Float32),   &requantConvertFunc<uint16_t, float>    },
    // Float32
    { TypePair(QnnDataType::Float32,  QnnDataType::UFixed8),   &requantConvertFunc<float,     uint8_t>  },
    { TypePair(QnnDataType::Float32,  QnnDataType::UFixed16),  &requantConvertFunc<float,     uint16_t> },
    { TypePair(QnnDataType::Float32,  QnnDataType::SFixed8),   &requantConvertFunc<float,     int8_t>   },
    { TypePair(QnnDataType::Float32,  QnnDataType::SFixed16),  &requantConvertFunc<float,     int16_t>  },
    { TypePair(QnnDataType::Float32,  QnnDataType::Float32),   &requantConvertFunc<float,     float>    },
  };
  return m;
}

void Accumulator::setVisionParam(uint32_t visionPos,
                                 uint32_t temporal,
                                 uint32_t height,
                                 uint32_t width) {
  visionParam.push_back(visionPos);
  visionParam.push_back(temporal);
  visionParam.push_back(height);
  visionParam.push_back(width);
}

void Accumulator::requantEmbedding(
    void* src, std::string srcDataType, double srcScale, int32_t srcOffset, size_t length) {
  double requantScale    = srcScale / scale;
  int32_t requantOffset  = srcOffset * requantScale - offset;
  size_t dstStartAddress = embeddingsBuffer.size() / byteWidth;
  size_t embeddingSize   = length * byteWidth;
  embeddingsBuffer.resize(embeddingsBuffer.size() + embeddingSize);
  void* dst = embeddingsBuffer.data();

  QnnDataType srcType = toQnnDataType(srcDataType);
  QnnDataType dstType = toQnnDataType(dataType);
  const auto& converterMap = Accumulator::getConverterMap();
  TypePair tp{srcType, dstType};
  auto it = converterMap.find(tp);
  if (it == converterMap.end()) {
    throw Exception(GENIE_STATUS_ERROR_GENERAL, "unsupported requant operation");
  }
  // Unified call
  it->second(src, dst, dstStartAddress, length, requantScale, requantOffset, srcScale, srcOffset);
}