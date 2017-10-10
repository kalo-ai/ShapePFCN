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

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/crf_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/math_functions.hpp"
#include <omp.h>

namespace caffe {

  template <typename Dtype>
  void CRFLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
  {
    if (this->layer_param_.loss_weight_size() == 0) 
      this->layer_param_.add_loss_weight(Dtype(1));

    CHECK_EQ(bottom[0]->num(), bottom[1]->num()) << "The unary data and pairwise data should originate from the same number of meshes in the batch [probably you should use a 'SoftmaxWithLoss' if pairwise data are empty.";
    CHECK_EQ(bottom[1]->num(), bottom[2]->num()) << "Label data and unary/pairwse data should originate from the same number of meshes in the batch.";
    CHECK_EQ(bottom[0]->height(), bottom[2]->height()) << "Number of max faces should be consistent for label and unary data";
    CHECK_EQ(bottom[0]->width(), 1) << "Expected blob width=1 for unary data!";
    CHECK_EQ(bottom[1]->width(), 1) << "Expected blob width=1 for pairwise data!";
    CHECK_EQ(bottom[2]->width(), 1) << "Expected blob width=1 for label data!";
    CHECK_GT(bottom[1]->height(), 1) << "Pairwise features empty!"; //size 1 means no pairwise features or max pairwise entries were set to 0 in data layer

    label_map_size = bottom[0]->channels();
    max_num_faces = bottom[0]->height(); // inner_num_ before
    max_num_pairwise_entries = bottom[1]->height();
    num_meshes = bottom[0]->num(); //outer_num_ before
    has_ignore_label_ = this->layer_param_.crf_loss_param().has_ignore_label();
    if (has_ignore_label_)
    {
      ignore_label_ = this->layer_param_.crf_loss_param().ignore_label();
    }

    pretrained_parms_file = this->layer_param_.crf_loss_param().pretrained_parms_file();
    CHECK(loadCRFParameters()) << "Could not load " << pretrained_parms_file;
    num_pairwise_kernels = pairwise_kernel_weights.rows;
    DLOG(INFO) << "num_pairwise_kernels (#pairwise features)=" << num_pairwise_kernels;
    CHECK_GT(num_pairwise_kernels, 0) << "Found 0 pairwise kernels (features) in input parameters file";
    learning_rate_for_pairwise_parameters = this->layer_param_.crf_loss_param().learning_rate_for_pairwise_parameters();
    DLOG(INFO) << "learning_rate_for_pairwise_parameters=" << learning_rate_for_pairwise_parameters;
    CHECK_GE(learning_rate_for_pairwise_parameters, 0) << "Learning rate for crf pairwise parameters should be positive or 0";
    delta_pairwise_label_incompatibility = cv::Mat::zeros(label_map_size, label_map_size, CV_32F);
    delta_pairwise_kernel_weights = cv::Mat::zeros(pairwise_kernel_weights.rows, 1, CV_32F);
  }

  template <typename Dtype>
  void CRFLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
  {
    vector<int> loss_shape(0);  // Loss layers output a scalar; 
    top[0]->Reshape(loss_shape);
    if (top.size() >= 2)
    {
      top[1]->ReshapeLike(*bottom[0]);
    }
    prob_.ReshapeLike(*bottom[0]);
  }


  template <typename Dtype>
  void CRFLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
  {
    Dtype* prob_data = prob_.mutable_cpu_data();
    caffe_set(prob_.count(), Dtype(0.0), prob_data);
    for (size_t m = 0; m < num_meshes; ++m)
      mfinference(bottom, m, 10, false); // run MF 10 iterations, outputs prob_

    const Dtype* label = bottom[2]->cpu_data();
    int dim = prob_.count() / num_meshes;
    int count = 0;
    Dtype loss = 0;
    for (int i = 0; i < num_meshes; ++i)
    {
      for (int j = 0; j < max_num_faces; j++)
      {
        const int label_value = static_cast<int>(label[i * max_num_faces + j]);
        if (has_ignore_label_ && label_value == ignore_label_)
          continue;
        DCHECK_GE(label_value, 0);
        DCHECK_LT(label_value, prob_.shape(1));
        loss -= log(std::max(prob_data[i * dim + label_value * max_num_faces + j], Dtype(FLT_MIN)));
        ++count;
      }
    }
    Dtype norm = (Dtype)count;
    if (norm < 1e-8)
      loss = 0;
    else
      loss = loss / norm;

    top[0]->mutable_cpu_data()[0] = loss;
    if (top.size() == 2)
      top[1]->ShareData(prob_);
  }

  template <typename Dtype>
  void CRFLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
  {
    train(bottom);

    if (propagate_down[1] || propagate_down[2])
    {
      LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs or  CRF pairwise features.";
    }
    if (propagate_down[0])
    {
      Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
      const Dtype* prob_data = prob_.cpu_data();
      caffe_copy(prob_.count(), prob_data, bottom_diff);
      const Dtype* label = bottom[2]->cpu_data();
      int dim = prob_.count() / num_meshes;
      int count = 0;
      for (int i = 0; i < num_meshes; ++i)
      {
        for (int j = 0; j < max_num_faces; ++j)
        {
          const int label_value = static_cast<int>(label[i * max_num_faces + j]);
          if (has_ignore_label_ && label_value == ignore_label_)
          {
            for (int c = 0; c < label_map_size; ++c)
            {
              bottom_diff[i * dim + c * max_num_faces + j] = 0;
            }
          }
          else
          {
            bottom_diff[i * dim + label_value * max_num_faces + j] -= 1;
            ++count;
          }
        }
      }
      // Scale gradient
      Dtype norm = Dtype(count);
      if (norm > 1e-8)
      {
        Dtype loss_weight = top[0]->cpu_diff()[0] / norm;
        caffe_scal(prob_.count(), loss_weight, bottom_diff);
      }
    }
  }

  template <typename Dtype>
  bool CRFLossLayer<Dtype>::loadCRFParameters()
  {
    cv::Mat pairwise_label_incompatibility_tmp;
    cv::Mat pairwise_kernel_weights_tmp;

    try
    {
      cv::FileStorage crf_parameter_file(pretrained_parms_file, cv::FileStorage::READ);
      crf_parameter_file["pairwise_label_incompatibility"] >> pairwise_label_incompatibility_tmp;
      crf_parameter_file["pairwise_kernel_weights"] >> pairwise_kernel_weights_tmp;
      crf_parameter_file.release();
      if (!pairwise_label_incompatibility_tmp.data)
        return false;
      if (!pairwise_kernel_weights_tmp.data)
        return false;
    }
    catch (cv::Exception& e__)
    {
      LOG(INFO) << e__.what();
      return false;
    }
    pairwise_label_incompatibility = pairwise_label_incompatibility_tmp.clone();
    pairwise_kernel_weights = pairwise_kernel_weights_tmp.clone();
    LOG(INFO) << "Imported pre-trained CRF parameters.";
    DLOG(INFO) << "pairwise_label_incompatibility:\n" << pairwise_label_incompatibility;
    DLOG(INFO) << "pairwise_kernel_weights:\n" << pairwise_kernel_weights;
    return true;
  }


  template <typename Dtype>
  bool CRFLossLayer<Dtype>::outputCRFParameters()
  {
    try
    {
      cv::FileStorage crf_parameter_file(pretrained_parms_file, cv::FileStorage::WRITE);
      crf_parameter_file << "pairwise_label_incompatibility" << pairwise_label_incompatibility;
      crf_parameter_file << "pairwise_kernel_weights" << pairwise_kernel_weights;
      crf_parameter_file.release();
    }
    catch (cv::Exception& e__)
    {
      LOG(INFO) << e__.what();
      return false;
    }
    return true;
  }


  template <typename Dtype>
  void CRFLossLayer<Dtype>::train(const vector<Blob<Dtype>*>& bottom)
  {
    if (learning_rate_for_pairwise_parameters <= 1e-8f)
      return;

    // run one iteration of training [completely outside caffe]
    cv::Mat derivative_pairwise_kernel_weights = cv::Mat::zeros(pairwise_kernel_weights.rows, 1, CV_32F);
    cv::Mat derivative_pairwise_label_incompatibility = cv::Mat::zeros(label_map_size, label_map_size, CV_32F);

    for (size_t m = 0; m < num_meshes; ++m)
    {
      mfinference(bottom, m, 10, true); // optimize for MF 10 iterations, otherwise too slow, will also populate derivatives for that mesh
      derivative_pairwise_label_incompatibility += mesh_derivative_pairwise_label_incompatibility;
      derivative_pairwise_kernel_weights += mesh_derivative_pairwise_kernel_weights;
    }  // end of loop over meshes

    // normalize
    derivative_pairwise_label_incompatibility /= float(num_meshes);
    derivative_pairwise_kernel_weights /= float(num_meshes);

    // compute delta of parameters
    //delta_pairwise_label_incompatibility = .9f * delta_pairwise_label_incompatibility + learning_rate_for_pairwise_parameters * derivative_pairwise_label_incompatibility;
    //delta_pairwise_kernel_weights = .9f * delta_pairwise_kernel_weights + learning_rate_for_pairwise_parameters * derivative_pairwise_kernel_weights;
    delta_pairwise_label_incompatibility = learning_rate_for_pairwise_parameters * derivative_pairwise_label_incompatibility;
    delta_pairwise_kernel_weights = learning_rate_for_pairwise_parameters * derivative_pairwise_kernel_weights;

    // limit aggressive changes
    delta_pairwise_label_incompatibility = cv::max(cv::min(delta_pairwise_label_incompatibility, 1.0f), -1.0f);
    delta_pairwise_kernel_weights = cv::max(cv::min(delta_pairwise_kernel_weights, 1.0f), -1.0f);

    // update parameters
    pairwise_label_incompatibility += delta_pairwise_label_incompatibility;
    pairwise_kernel_weights += delta_pairwise_kernel_weights;

    // projected gradient ascent
    pairwise_label_incompatibility.setTo(1e-5f, pairwise_label_incompatibility < 1e-5f);
    pairwise_kernel_weights.setTo(1e-5f, pairwise_kernel_weights < 1e-5f);

    // stats
    LOG(INFO) << "Label Compatibility Parameters: ";
    LOG(INFO) << pairwise_label_incompatibility;
    LOG(INFO) << "Kernel weights: ";
    LOG(INFO) << pairwise_kernel_weights;

    CHECK(outputCRFParameters()) << "Could not update crf parameters file " << pretrained_parms_file;
  }



  template <typename Dtype>
  void CRFLossLayer<Dtype>::mfinference(const vector<Blob<Dtype>*>& bottom, int mesh_id, int max_iter, bool used_for_learning)
  {
    // initialize mf and related probabilities
    int unary_data_offset = mesh_id * (prob_.count() / num_meshes);
    int pairwise_data_offset = mesh_id * max_num_pairwise_entries;
    int label_data_offset = mesh_id * max_num_faces;

    const Dtype* unary_data = bottom[0]->cpu_data() + unary_data_offset;
    Dtype* prob_data = prob_.mutable_cpu_data() + unary_data_offset;
    const Dtype* pairwise_data = bottom[1]->cpu_data() + pairwise_data_offset;
    const Dtype* label_data = bottom[2]->cpu_data() + label_data_offset;

    int number_of_faces = 0;
    for (int j = 0; j < max_num_faces; ++j)
    {
      if ((int)label_data[j] == 255) // if labeling is incomplete, this would be a problem [need another ignore label e.g., 254]
        break;
      number_of_faces++;
    }
    DLOG(INFO) << "Number of faces for this mesh: " << number_of_faces;
    cv::Mat face_log_unary_features(number_of_faces, label_map_size, CV_32F);
    cv::Mat face_unary_probabilities(number_of_faces, label_map_size, CV_32F);    
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int j = 0; j < number_of_faces; ++j)
    {
      for (int c = 0; c < label_map_size; ++c)
      {
        face_log_unary_features.at<float>(j, c) = unary_data[c * max_num_faces + j];
      }
    }

    cv::Mat logQ = face_log_unary_features.clone();
    cv::Mat row_sum(number_of_faces, 1, CV_32F);
    cv::Mat row_max(number_of_faces, 1, CV_32F);

    cv::reduce(face_log_unary_features, row_max, 1, CV_REDUCE_MAX);
    for (int i = 0; i < face_log_unary_features.rows; ++i)
      face_log_unary_features.row(i) -= row_max.at<float>(i);
    cv::exp(face_log_unary_features, face_unary_probabilities);

    cv::reduce(face_unary_probabilities, row_sum, 1, CV_REDUCE_SUM);
    for (int i = 0; i < face_unary_probabilities.rows; ++i)
      face_unary_probabilities.row(i) /= row_sum.at<float>(i);

    //cv::Mat thr_unary_probabilities = face_unary_probabilities.clone();
    //thr_unary_probabilities.setTo(.00001f, thr_unary_probabilities < .00001f); // avoid numerical problems, overconfident unary features
    //thr_unary_probabilities.setTo(.99999f, thr_unary_probabilities > .99999f);
    //cv::log(thr_unary_probabilities, face_log_unary_features);

    cv::Mat face_mf_probabilities = face_unary_probabilities.clone();

    // save all nodes so that we use openmp (<3.0v)
    vector< vector< std::tuple<int, int, float> > > nodes(number_of_faces);
    float counts = 0.0f;
    for (int fpe = 0; fpe < max_num_pairwise_entries; fpe += 3)
    {
      if ((int)pairwise_data[fpe] == -1)
        break;
      counts++;
      int feature_id = (int)pairwise_data[fpe] / max_num_faces;
      int face_id1 = (int)pairwise_data[fpe] % max_num_faces;
      int face_id2 = (int)pairwise_data[fpe + 1];
      float dissimilarity = pairwise_data[fpe + 2];
      nodes[face_id1].push_back(std::make_tuple(feature_id, face_id2, dissimilarity));
    }


#if defined(_OPENMP)
    int num_threads_to_use = omp_get_max_threads();
    omp_set_num_threads(num_threads_to_use);
#else
    int num_threads_to_use = 1;
#endif

    // MF iterations (must be fast!)
    for (int mf_iter = 1; mf_iter <= max_iter; ++mf_iter)
    {
      logQ = face_log_unary_features.clone();

#if defined(_OPENMP)
#pragma omp parallel for
#endif
      for (int node = 0; node < number_of_faces; ++node)
      {
        for (unsigned int adj_node = 0; adj_node < nodes[node].size(); ++adj_node)
        {
          std::tuple<int, int, float>& entry = nodes[node][adj_node];
          float dissimilarity = std::get<2>(entry);
          float kernel_weight = pairwise_kernel_weights.at<float>(std::get<0>(entry));

          for (int c2 = 0; c2 < label_map_size; ++c2)
          {
            float nbr_mf_prob = face_mf_probabilities.at<float>(std::get<1>(entry), c2);
            float nbr_mf_prob_mult_kernel = nbr_mf_prob * kernel_weight;

            for (int c = 0; c < label_map_size; ++c)
            {
              if (c != c2)
              {
                logQ.at<float>(node, c) -= nbr_mf_prob_mult_kernel * pairwise_label_incompatibility.at<float>(c, c2) * (1.0f - dissimilarity);
                //logQ.at<float>(n->idx[1], c) -= nbr_mf_prob_mult_kernel * pairwise_label_incompatibility.at<float>(c, c2) * exp(-dissimilarity); // more like previous dense crfs
              }
              else
              {
                logQ.at<float>(node, c) -= nbr_mf_prob_mult_kernel * pairwise_label_incompatibility.at<float>(c, c) * dissimilarity;
              }
            } // end of loop over c 
          } // end of loop over c2
        } // end of loop over adj_nodes
      } // end of loop over faces [end of parallelism]

      // scale Q to avoid numerical explosion
      cv::reduce(logQ, row_max, 1, CV_REDUCE_MAX);
      for (int i = 0; i < logQ.rows; ++i)
        logQ.row(i) -= row_max.at<float>(i);
      cv::exp(logQ, face_mf_probabilities);
      cv::reduce(face_mf_probabilities, row_sum, 1, CV_REDUCE_SUM);
      for (int i = 0; i < face_mf_probabilities.rows; ++i)
        face_mf_probabilities.row(i) /= row_sum.at<float>(i);
    }

    if (!used_for_learning)
    {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
      for (int j = 0; j < number_of_faces; ++j)
      {
        for (int c = 0; c < label_map_size; ++c)
        {
          prob_data[c * max_num_faces + j] = face_mf_probabilities.at<float>(j, c);
        }
      }
    }
    else
    {
      //// some corrections might be impossible without 'smoothing' the probs
      face_mf_probabilities += 1e-5f; // equivalent to 'robust' log-likelihood
      cv::reduce(face_mf_probabilities, row_sum, 1, CV_REDUCE_SUM);
      for (int i = 0; i < face_mf_probabilities.rows; ++i)
        face_mf_probabilities.row(i) /= row_sum.at<float>(i);

      mesh_derivative_pairwise_label_incompatibility = cv::Mat::zeros(label_map_size, label_map_size, CV_32F);
      mesh_derivative_pairwise_kernel_weights = cv::Mat::zeros(pairwise_kernel_weights.rows, 1, CV_32F);
      vector< cv::Mat > mesh_derivative_pairwise_label_incompatibility_threads(num_threads_to_use);
      vector< cv::Mat > mesh_derivative_pairwise_kernel_weights_threads(num_threads_to_use);
      for (int tid = 0; tid < num_threads_to_use; tid++)
      {
        mesh_derivative_pairwise_label_incompatibility_threads[tid] = cv::Mat::zeros(label_map_size, label_map_size, CV_32F);
        mesh_derivative_pairwise_kernel_weights_threads[tid] = cv::Mat::zeros(pairwise_kernel_weights.rows, 1, CV_32F);
      }

#if defined(_OPENMP)
#pragma omp parallel for
#endif
      for (int node = 0; node < number_of_faces; ++node)
      {
        for (unsigned int adj_node = 0; adj_node < nodes[node].size(); ++adj_node)
        {
          std::tuple<int, int, float>& entry = nodes[node][adj_node];
          int ground_truth_label_n = (int)label_data[node];
          int ground_truth_label_nbr = (int)label_data[std::get<1>(entry)];
          float dissimilarity = std::get<2>(entry);
          float kernel_weight = pairwise_kernel_weights.at<float>(std::get<0>(entry));
#if defined(_OPENMP)
          int tid = omp_get_thread_num();
#else
          int tid = 0;
#endif

          // data-dependent gradient terms
          if (ground_truth_label_n != ground_truth_label_nbr)
          {
            mesh_derivative_pairwise_label_incompatibility_threads[tid].at<float>(ground_truth_label_n, ground_truth_label_nbr) -= kernel_weight * (1.0f - dissimilarity);
            mesh_derivative_pairwise_kernel_weights_threads[tid].at<float>(std::get<0>(entry)) -= pairwise_label_incompatibility.at<float>(ground_truth_label_n, ground_truth_label_nbr) * (1.0f - dissimilarity);
          }
          else
          {
            mesh_derivative_pairwise_label_incompatibility_threads[tid].at<float>(ground_truth_label_n, ground_truth_label_n) -= kernel_weight * dissimilarity;
            mesh_derivative_pairwise_kernel_weights_threads[tid].at<float>(std::get<0>(entry)) -= pairwise_label_incompatibility.at<float>(ground_truth_label_n, ground_truth_label_n) * dissimilarity;
          }

          // model-dependent gradient terms
          for (int c2 = 0; c2 < label_map_size; ++c2)
          {
            float nbr_mf_prob = face_mf_probabilities.at<float>(std::get<1>(entry), c2);

            for (int c = 0; c < label_map_size; ++c)
            {
              float n_mf_prob = face_mf_probabilities.at<float>(std::get<1>(entry), c);

              if (c != c2)
              {
                float mf_probs_mult_dissimilarity = n_mf_prob * nbr_mf_prob * (1.0f - dissimilarity);
                mesh_derivative_pairwise_label_incompatibility_threads[tid].at<float>(c, c2) += kernel_weight * mf_probs_mult_dissimilarity;
                mesh_derivative_pairwise_kernel_weights_threads[tid].at<float>(std::get<0>(entry)) += pairwise_label_incompatibility.at<float>(c, c2) * mf_probs_mult_dissimilarity;
              }
              else
              {
                float mf_probs_mult_dissimilarity = n_mf_prob * nbr_mf_prob * dissimilarity;
                mesh_derivative_pairwise_label_incompatibility_threads[tid].at<float>(c, c) += kernel_weight * mf_probs_mult_dissimilarity;
                mesh_derivative_pairwise_kernel_weights_threads[tid].at<float>(std::get<0>(entry)) += pairwise_label_incompatibility.at<float>(c, c) * mf_probs_mult_dissimilarity;
              }
            }
          }
        } // end of loop over adj faces per face
      } // end of loop over faces (enf of parallelism)

      for (int tid = 0; tid < num_threads_to_use; ++tid)
      {
        mesh_derivative_pairwise_label_incompatibility += mesh_derivative_pairwise_label_incompatibility_threads[tid];
        mesh_derivative_pairwise_kernel_weights += mesh_derivative_pairwise_kernel_weights_threads[tid];
      }
      mesh_derivative_pairwise_label_incompatibility = mesh_derivative_pairwise_label_incompatibility / counts;
      mesh_derivative_pairwise_kernel_weights = mesh_derivative_pairwise_kernel_weights / counts;
    } // end of if condition for learning
  }


  INSTANTIATE_CLASS(CRFLossLayer);
  REGISTER_LAYER_CLASS(CRFLoss);

}  // namespace caffe
