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

#ifndef CAFFE_CRF_LOSS_LAYER_HPP_
#define CAFFE_CRF_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace caffe {

  /**
  * @brief Computes the mesh CRF-based loss function (see paper)
  *  
  * @param bottom input Blob vector (length 3)
  *   -# @f$ ([batch size] \times [num labels] \times [max num faces] \times 1) @f$
  *      predictions (unary term): a Blob with values in @f$ [-\infty, +\infty] @f$ indicating the predicted score 
         for each of the classes per mesh face. This layer maps these scores to a
  *      probability distribution over classes using the CRF
  *   -# @f$ ([batch size] \times 1 \times [max pairwise entries] \times 1) @f$
  *      CRF pairwise features: a flattened vector storing all CRF pairwise features per face pair
  *   -# @f$ ([batch size] \times 1 \times [max num faces] \times 1) @f$
  *      labels: an integer-valued Blob with values @f$ l_n \in [0, 1, 2, ..., K - 1] @f$
  *      indicating the correct class label per face. Last entry stores #mesh faces
  * @param top output Blob vector (length 1)
  *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
  *      the computed CRF loss function 
  *   -# @f$ ([batch size] \times [num labels] \times [max num faces] \times 1)  @f$
  *      CRF probabilities (based on mean-field)
  */
  template <typename Dtype>
  class CRFLossLayer : public LossLayer<Dtype> {
  public:
    /**
    * @param param provides CRFLossParameter crf_loss_param, with options:
    *  - ignore_label (optional)
    *    Specify a label value that should be ignored when computing the loss.
    */
    explicit CRFLossLayer(const LayerParameter& param): LossLayer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "CRFLoss"; }
    virtual inline int ExactNumBottomBlobs() const { return 3; }
    virtual inline int ExactNumTopBlobs() const { return -1; }
    virtual inline int MinTopBlobs() const { return 1; }
    virtual inline int MaxTopBlobs() const { return 2; }

    /**
    * We usually cannot backpropagate to the labels and pairwise features; ignore force_backward for these inputs.
    */
    virtual inline bool AllowForceBackward(const int bottom_index) const {
      return ( (bottom_index != 1) & (bottom_index != 2) );
    }

  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    /**
    * @brief Computes the CRF error gradient w.r.t. the predictions.
    *
    * Gradients cannot be computed with respect to the pairwise feature inputs (bottom[1]),
    * so this method ignores bottom[1] and requires !propagate_down[1], crashing
    * if propagate_down[1] is set.
    * Gradients cannot be computed with respect to the label inputs (bottom[2]),
    * so this method ignores bottom[2] and requires !propagate_down[2], crashing
    * if propagate_down[2] is set.
    *
    * @param top output Blob vector (length 1), providing the error gradient with
    *      respect to the outputs
    *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
    *      This Blob's diff will simply contain the loss_weight* @f$ \lambda @f$,
    *      as @f$ \lambda @f$ is the coefficient of this layer's output
    *      @f$\ell_i@f$ in the overall Net loss
    *      @f$ E = \lambda_i \ell_i + \mbox{other loss terms}@f$; hence
    *      @f$ \frac{\partial E}{\partial \ell_i} = \lambda_i @f$.
    *      (*Assuming that this top Blob is not used as a bottom (input) by any
    *      other layer of the Net.)
    * @param propagate_down see Layer::Backward.
    *      propagate_down[1] and propagate_down[2] must be false as we can't compute gradients with
    *      respect to the labels and pairwise features.
    * @param bottom input Blob vector (length 3)
    *   -# @f$ ([batch size] \times [num labels] \times [max num faces] \times 1) @f$
    *      the predictions @f$ x @f$; Backward computes diff @f$ \frac{\partial E}{\partial x} @f$
    *   -# @f$ ([batch size] \times 1 \times [max pairwise entries] \times 1) @f$
    *      CRF pairwise features -- ignored as we can't compute their error gradients
    *   -# @f$ ([batch size] \times 1 \times [max num faces] \times 1) @f$
    *      labels -- ignored as we can't compute their error gradients
    */
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    /// Whether to ignore instances with a certain label.
    bool has_ignore_label_;
    /// The label indicating that an instance should be ignored.
    int ignore_label_;

    /// prob stores the output probability predictions from the CRF.
    Blob<Dtype> prob_;

    // adapted CRF machinery (see MeshProcessor.cpp)
    int label_map_size;
    int max_num_faces;  // inner_num_ before
    int max_num_pairwise_entries;
    int num_meshes; //outer_num_ before
    float learning_rate_for_pairwise_parameters;
    string pretrained_parms_file;
    int num_pairwise_kernels;
    cv::Mat pairwise_label_incompatibility;
    cv::Mat pairwise_kernel_weights;
    cv::Mat mesh_derivative_pairwise_kernel_weights;
    cv::Mat mesh_derivative_pairwise_label_incompatibility;
    cv::Mat delta_pairwise_kernel_weights;
    cv::Mat delta_pairwise_label_incompatibility;

    bool loadCRFParameters();
    bool outputCRFParameters();

    void train(const vector<Blob<Dtype>*>& bottom);
    void mfinference(const vector<Blob<Dtype>*>& bottom, int mesh_id, int max_iter = 10, bool used_for_learning = false);
  };


}  // namespace caffe


#endif  // CAFFE_CRF_LOSS_LAYER_HPP_
