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

#ifndef __MVFCN_FCNShapes_hpp__
#define __MVFCN_FCNShapes_hpp__

#include "Common.hpp"
#include "MeshProcessor.hpp"
#include "RenderViews.hpp"
#ifndef SKIP_COMPILING_CAFFE_NETWORK_CODE
#include <caffe/caffe.hpp>
#include "google/protobuf/text_format.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "MeshCRF.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

//const static cv::Scalar mean = cv::Scalar(102.93f, 111.36f, 116.52f, 0.0);
const static float image_mean = 110.27;
const static float depth_mean = 127.5;
const static float aux_mean = 127.5; // SDF/UP change
const static int label_margin = 186;

#endif

class MVFCN
{
private:
  int num_camera_orbits;
  std::vector< std::shared_ptr<MeshProcessor> > meshes_processor_ptr;
  std::map<string, int> label_map;
  std::map<string, bool> validation_map;
  typedef std::map<string, int>::const_iterator label_map_iterator;
  typedef std::map<string, bool>::const_iterator validation_map_iterator;

  vector<int> gpus;
  vector<float> mesh_labeling_accuracies;
  vector<float> state;
  bool google_logging_is_initialized;
  ViewPoolingOperator view_pooling_type;

#ifndef SKIP_COMPILING_CAFFE_NETWORK_CODE  
  std::shared_ptr<MeshCRF> crf;

  void fcntrain(const size_t num_classes, const string& train_dataset_path, bool include_validation_data_for_training = false);
  void mvfcntrain(const size_t num_classes, const string& train_dataset_path, const bool use_crf_mvfcn);
  bool fcntest(const size_t num_classes, const string& train_dataset_path, const string& test_dataset_path, const bool use_mvfcn_model, const bool do_not_check_views = false);
  void get_gpus();
  void outputMeshLabelingAccuracies(const string& dataset_path, const bool used_mvfcn_model, const bool called_from_training);
  void loadState(const string& state_filename);
  void saveState(const string& state_filename);
  string findLatestShapshot(const string& search_path, const string& snapshot_base_name);
  void deleteShapshots(const string& search_path, const string& snapshot_base_name);
#endif
  bool createAuxiliaryDirectories(const string& dataset_path, const bool skip_rendering_directories);
  

public:
  MVFCN(ViewPoolingOperator _view_pooling_type = MAX_VIEW_POOLING);
  void train();
  void test();
  void setPoolingType(ViewPoolingOperator _view_pooling_type) { view_pooling_type = _view_pooling_type; }
};
#endif