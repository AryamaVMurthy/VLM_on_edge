//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once
#ifndef IMAGE_ENCODER_HPP
#define IMAGE_ENCODER_HPP
#include <memory>
#include <vector>

#include "GeniePipeline.h"
#include "Node.hpp"

namespace genie {
namespace pipeline {

class Pipeline;

class ImageEncoder final : public Node {
 public:
  ImageEncoder(qualla::json config,
               std::shared_ptr<ProfileStat> profileStat,
               std::shared_ptr<genie::log::Logger> logger = nullptr);

  int32_t execute(void* userData, std::shared_ptr<ProfileStat> profileStat);
  int32_t setImageInputData(GenieNode_IOName_t nodeIOName,
                            const void* imageData,
                            size_t imageSize,
                            std::shared_ptr<ProfileStat> profileStat);
  int32_t setEmbeddingOutputCallback(GenieNode_IOName_t nodeIOName,
                                     GenieNode_EmbeddingOutputCallback_t callback);

  int32_t applyLora(std::string loraAdapterName,
                    std::string engine,
                    std::shared_ptr<ProfileStat> profileStat);
  int32_t applyLoraStrength(std::string tensorName, std::string engine, float alpha);

 private:
  std::shared_ptr<Embedding> m_encoder;
  std::vector<uint8_t> m_data;
  std::unordered_map<std::string, std::vector<uint8_t>> m_input;
  std::unordered_map<GenieNode_IOName_t, std::string> m_inputIOMap;
  GenieNode_EmbeddingOutputCallback_t m_embeddingOutputCallback;

  uint32_t m_height{0};
  uint32_t m_width{0};
};

}  // namespace pipeline
}  // namespace genie
#endif  // IMAGE_ENCODER_HPP