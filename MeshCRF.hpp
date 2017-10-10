//============================================================================
//
// This file is part of the ShapePFCN project.
//
// Copyright (c) 2016-2017 - Evangelos Kalogerakis, Melinos Averkiou, Siddhartha Chaudhuri, Subhransu Maji
//
// ShapePFCN is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// ShapePFCN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with ShapePFCN.  If not, see <http://www.gnu.org/licenses/>.
//
//============================================================================

#ifndef __MVFCN_MeshCRF_hpp__
#define __MVFCN_MeshCRF_hpp__

#include "Common.hpp"
#include "MeshProcessor.hpp"
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <omp.h>


#ifndef SKIP_COMPILING_CAFFE_NETWORK_CODE
class MeshCRF
{
private:
  bool called_from_joint_mvfcn_model;
  std::map<string, int> label_map;
  string crf_input_output_folder;
  cv::Mat pairwise_label_incompatibility;
  cv::Mat pairwise_kernel_weights;
  cv::Mat mesh_derivative_pairwise_kernel_weights;
  cv::Mat mesh_derivative_pairwise_label_incompatibility;

  unsigned long long getMemAvailable();

public:
  MeshCRF(): called_from_joint_mvfcn_model(false) {};
  MeshCRF(const std::map<string, int>& _label_map, const string& _input_output_folder, const int num_pairwise_kernels);

  void train(const std::vector< std::shared_ptr<MeshProcessor> >& meshes_processor_ptr, const string& output_parameter_filename);
  bool loadCRFParameters(const string& input_parameter_filename);
  bool outputCRFParameters(const string& output_parameter_filename);
  float mfinference(const std::shared_ptr<MeshProcessor>& mesh_processor_ptr, int max_iter = 10, bool used_for_learning = false);
  void isItCalledFromJointMVFCNModel(bool called_from_joint_mvfcn_model_) { called_from_joint_mvfcn_model = called_from_joint_mvfcn_model_; }
};
#endif

#endif