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

// mostly based on Fisher Yu's "Multi-Scale Context Aggregation by Dilated Convolutions" image data layer

#ifndef CAFFE_IMAGE_DEPTH_LABEL_DATA_LAYER_H
#define CAFFE_IMAGE_DEPTH_LABEL_DATA_LAYER_H

#include <random>
#include <vector>
#include <iterator>
#include <algorithm>
#include <fstream>   // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <regex>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

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
class ImageDepthLabelDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
   explicit ImageDepthLabelDataLayer(const LayerParameter &param);

   virtual ~ImageDepthLabelDataLayer();

  virtual void DataLayerSetUp(const vector<Blob<Dtype> *> &bottom,
                              const vector<Blob<Dtype> *> &top);

  // DataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }

  virtual inline const char *type() const { return "ImageDepthLabelData"; }

  virtual inline int ExactNumBottomBlobs() const { return 0; }

  virtual inline int ExactNumTopBlobs() const { return 2; }

  virtual inline int MaxTopBlobs() const { return 2; }

  virtual inline int MinTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;

  virtual void load_batch(Batch<Dtype>* batch);

  virtual void ShuffleImages(vector<string>& rendered_image_filenames);
  
  virtual cv::Mat ExtendLabelMargin(cv::Mat &image, int margin_w, int margin_h, double value = -1.0);   

  virtual vector<string> searchForImages(const string& search_path);

  void Transform(const cv::Mat& cv_img, const cv::Mat& cv_depth_img, const cv::Mat& cv_aux_img, const cv::Mat& cv_label_img, cv::Mat& cv_transformed_img, cv::Mat& cv_transformed_label_img); // SDF/UP change

  string rendered_image_dir_;
  string depth_image_dir_;
  string aux_image_dir_; // SDF/UP change
  string label_dir_;
  int image_margin_;
  int batch_size_;
  int validation_mode_;

  vector<std::string> image_filename_lines_;
  int image_id_;
  int crop_size_;
  int crop_minus_margin_size_;  

  float image_mean_;
  float depth_mean_;
  float aux_mean_; // SDF/UP change

  vector<int> data_shape;
  vector<int> label_shape;

  std::mt19937 *rng_;
};

} // namspace caffe


#endif //CAFFE_IMAGE_DEPTH_LABEL_DATA_LAYER_H
