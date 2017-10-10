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

#ifndef CAFFE_MESH_IMAGE_LABEL_DATA_LAYER_H
#define CAFFE_MESH_IMAGE_LABEL_DATA_LAYER_H

#include <random>
#include <vector>
#include <iterator>
#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <regex>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include <boost/filesystem.hpp>
#include <boost/algorithm/string/replace.hpp>


namespace caffe {

template<typename Dtype>
class MeshImageLabelDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit MeshImageLabelDataLayer(const LayerParameter &param);

  virtual ~MeshImageLabelDataLayer();

  virtual void DataLayerSetUp(const vector<Blob<Dtype> *> &bottom,
                              const vector<Blob<Dtype> *> &top);

  // DataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }

  virtual inline const char *type() const { return "MeshImageLabelData"; }

  virtual inline int ExactNumBottomBlobs() const { return 0; }

  virtual inline int ExactNumTopBlobs() const { return 4; }

  virtual inline int MaxTopBlobs() const { return 4; }

  virtual inline int MinTopBlobs() const { return 4; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;

  virtual void load_batch(Batch<Dtype>* batch);

  virtual void ShuffleMeshes();

  virtual void ShuffleImagesOfMesh(vector<string>& rendered_image_filenames);
  
  virtual cv::Mat ExtendLabelMargin(cv::Mat &image, int margin_w, int margin_h, double value = -1.0);   

  virtual vector<string> searchForImagesOfAMesh(const string& search_path, const string &base_mesh_filename);

  void Transform(const cv::Mat& cv_img, const cv::Mat& cv_depth_img, const cv::Mat& cv_aux_img,  const cv::Mat& cv_tid_img, vector<cv::Mat>& cv_transformed_img, cv::Mat& cv_transformed_tid_img); // SDF/UP change

  string mesh_list_filename_;
  string rendered_image_dir_;
  string depth_image_dir_;
  string aux_image_dir_; // SDF/UP change
  string rendered_triangleid_dir_;
  string crf_features_dir_;
  int max_num_views_;
  int max_num_faces_;
  int max_num_pairwise_entries_;
  int num_pairwise_features_;
  int image_margin_;
  int batch_size_;

  vector<std::string> mesh_filename_lines_;
  int mesh_id_;
  int crop_size_;
  int crop_minus_margin_size_;  
  bool stochastic_;

  float image_mean_;
  float depth_mean_;
  float aux_mean_; // SDF/UP change

  vector<int> data_shape;
  vector<int> image2mesh_data_shape;
  vector<int> crfpairwise_data_shape;
  vector<int> label_shape;

  std::mt19937 *rng_;
};

} // namspace caffe


#endif //CAFFE_MESH_IMAGE_LABEL_DATA_LAYER_H
