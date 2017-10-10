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


#include "MeshCRF.hpp"

#ifndef SKIP_COMPILING_CAFFE_NETWORK_CODE

MeshCRF::MeshCRF(const std::map<string, int>& _label_map, const string& _input_output_folder, const int num_pairwise_kernels)
{
  called_from_joint_mvfcn_model = false;
  label_map = _label_map;
  crf_input_output_folder = _input_output_folder;
  pairwise_kernel_weights = cv::Mat::ones(num_pairwise_kernels, 1, CV_32F);
  pairwise_label_incompatibility = cv::Mat::ones((int)label_map.size(), (int)label_map.size(), CV_32F); 
}


unsigned long long MeshCRF::getMemAvailable()
{
#ifdef _WIN32
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  GlobalMemoryStatusEx(&status);
  return status.ullAvailPhys;
#else
  unsigned long long ps = sysconf(_SC_PAGESIZE);
  unsigned long long pn = sysconf(_SC_AVPHYS_PAGES);
  return ps * pn;
#endif

  return 0;
}

void MeshCRF::train(const std::vector< std::shared_ptr<MeshProcessor> >& meshes_processor_ptr, const string& output_parameter_filename)
{
  THEA_CONSOLE << "***** MESH CRF PRE-TRAINING STARTS HERE *****";

  // checking if avail mem is enough
  unsigned long long required_memory = 0;
  for (size_t m = 0; m < meshes_processor_ptr.size(); ++m)
  {    
    required_memory += (unsigned long long)meshes_processor_ptr[m]->number_of_entries_in_pairwise_features * sizeof(float);
    required_memory += (unsigned long long)meshes_processor_ptr[m]->number_of_faces * (unsigned long long)label_map.size() * sizeof(float);
  }
  unsigned long long available_memory = getMemAvailable();
  THEA_CONSOLE << "Required memory: " << required_memory << ", available memory: " << available_memory;
  bool delete_crf_mesh_data_every_iteration = false;
  if (required_memory > available_memory)
  {
    THEA_CONSOLE << "Unfortunately CRF training will be slow since all mesh CRF data do not fit in the main mem";
    delete_crf_mesh_data_every_iteration = true;
  }
  else
  {
    THEA_CONSOLE << "Excellent - there is enough memory to preload all mesh CRF data and be fast!";
  }

  THEA_CONSOLE << "Learning CRF Parameters...";
  THEA_CONSOLE << "Initial Label Compatibility Parameters: ";
  THEA_CONSOLE << pairwise_label_incompatibility;
  THEA_CONSOLE << "Initial Kernel weights: ";
  THEA_CONSOLE << pairwise_kernel_weights;
  float step = 10.0f;


  cv::Mat best_pairwise_label_incompatibility = pairwise_label_incompatibility.clone();
  cv::Mat best_pairwise_kernel_weights = pairwise_kernel_weights.clone();
  float   best_accuracy = 0.0f;
  int best_iteration = 0;
  int gradient_ascent_iter = 1;
  for (gradient_ascent_iter = 1; gradient_ascent_iter <= 1000; ++gradient_ascent_iter) 
  {
    // init for iteration
    THEA_CONSOLE << "Iteration " << gradient_ascent_iter << "...";
    float avg_accuracy = 0.0f;
    cv::Mat derivative_pairwise_kernel_weights = cv::Mat::zeros(pairwise_kernel_weights.rows, 1, CV_32F);
    cv::Mat derivative_pairwise_label_incompatibility = cv::Mat::zeros((int)label_map.size(), (int)label_map.size(), CV_32F);
    cv::Mat delta_pairwise_kernel_weights = cv::Mat::zeros(pairwise_kernel_weights.rows, 1, CV_32F);
    cv::Mat delta_pairwise_label_incompatibility = cv::Mat::zeros((int)label_map.size(), (int)label_map.size(), CV_32F);
    cv::Mat old_pairwise_kernel_weights = pairwise_kernel_weights.clone();
    cv::Mat old_pairwise_label_incompatibility = pairwise_label_incompatibility.clone();

    for (size_t m = 0; m < meshes_processor_ptr.size(); ++m)
    {
      // init for mesh
      shared_ptr<MeshProcessor> mesh_processor_ptr = meshes_processor_ptr[m];
      avg_accuracy += mfinference(mesh_processor_ptr, 10, true); // optimize for MF 10 iterations, otherwise too slow, wii also populate derivatives for that mesh

      derivative_pairwise_label_incompatibility += mesh_derivative_pairwise_label_incompatibility;
      derivative_pairwise_kernel_weights += mesh_derivative_pairwise_kernel_weights;

      if (delete_crf_mesh_data_every_iteration)
        mesh_processor_ptr->freeMeshCRFData(); // otherwise, we will run out of memory! 
    }  // end of loop over meshes

    // normalize
    avg_accuracy /= float(meshes_processor_ptr.size());
    derivative_pairwise_label_incompatibility /= float(meshes_processor_ptr.size());
    derivative_pairwise_kernel_weights /= float(meshes_processor_ptr.size());

    // keep current best parameters 
    if (avg_accuracy >= best_accuracy)
    {
      best_accuracy = avg_accuracy;
      best_pairwise_label_incompatibility = pairwise_label_incompatibility.clone();
      best_pairwise_kernel_weights = pairwise_kernel_weights.clone();
      best_iteration = gradient_ascent_iter - 1;
    }

    // compute delta of parameters
    //delta_pairwise_label_incompatibility = .9f * delta_pairwise_label_incompatibility + step * derivative_pairwise_label_incompatibility;
    //delta_pairwise_kernel_weights = .9f * delta_pairwise_kernel_weights + step * derivative_pairwise_kernel_weights;
    delta_pairwise_label_incompatibility = step * derivative_pairwise_label_incompatibility;
    delta_pairwise_kernel_weights = step * derivative_pairwise_kernel_weights;

      // limit aggressive changes
    delta_pairwise_label_incompatibility = cv::max(cv::min(delta_pairwise_label_incompatibility, 1.0f), -1.0f);
    delta_pairwise_kernel_weights = cv::max(cv::min(delta_pairwise_kernel_weights, 1.0f), -1.0f);

    // update parameters
    pairwise_label_incompatibility += delta_pairwise_label_incompatibility;
    pairwise_kernel_weights += delta_pairwise_kernel_weights;

    // projected gradient ascent
    pairwise_label_incompatibility.setTo(1e-5f, pairwise_label_incompatibility < 1e-5f);
    pairwise_kernel_weights.setTo(1e-5f, pairwise_kernel_weights < 1e-5f);

    // convergence?
    float diff = cv::sum(cv::abs(old_pairwise_label_incompatibility - pairwise_label_incompatibility))[0]
               + cv::sum(cv::abs(old_pairwise_kernel_weights - pairwise_kernel_weights))[0];
    diff /= float(pairwise_kernel_weights.total() + pairwise_label_incompatibility.total());
    step /= 1.01f;

    // stats
    THEA_CONSOLE << "Current [approximate] accuracy: " << avg_accuracy * 100.0f;
    THEA_CONSOLE << "Label Compatibility Parameters: ";
    THEA_CONSOLE << pairwise_label_incompatibility;
    THEA_CONSOLE << "Kernel weights: ";
    THEA_CONSOLE << pairwise_kernel_weights;
    THEA_CONSOLE << "Parameter total change: " << diff << ", step: " << step;

    if (diff < 1e-5f)
    {
      THEA_CONSOLE << "Converged (more or less).";
      break;
    }
  }


  float avg_accuracy = 0.0f;
  for (size_t m = 0; m < meshes_processor_ptr.size(); ++m)
  {
    shared_ptr<MeshProcessor> mesh_processor_ptr = meshes_processor_ptr[m];
    avg_accuracy += mfinference(mesh_processor_ptr, 10, true);
  }
  avg_accuracy /= float(meshes_processor_ptr.size());
  THEA_CONSOLE << "Final [approximate] accuracy: " << avg_accuracy * 100.0f;
  // keep current best parameters 
  if (avg_accuracy >= best_accuracy)
  {
    best_accuracy = avg_accuracy;
    best_pairwise_label_incompatibility = pairwise_label_incompatibility.clone();
    best_pairwise_kernel_weights = pairwise_kernel_weights.clone();
    best_iteration = gradient_ascent_iter;
  }

  // keeping only best iteration
  pairwise_label_incompatibility = best_pairwise_label_incompatibility.clone();
  pairwise_kernel_weights = best_pairwise_kernel_weights.clone();
  THEA_CONSOLE << "Final best [approximate] accuracy: " << best_accuracy * 100.0f << " (best iteration: " << best_iteration << ")";
  THEA_CONSOLE << "Final best label Compatibility Parameters: ";
  THEA_CONSOLE << pairwise_label_incompatibility;
  THEA_CONSOLE << "Final best kernel weights: ";
  THEA_CONSOLE << pairwise_kernel_weights;

  if (!outputCRFParameters(output_parameter_filename))
    THEA_ERROR << "Could not write crf parameters to " << output_parameter_filename << " - will use default non-optimized parameters during testing.";
}

bool MeshCRF::loadCRFParameters(const string& input_parameter_filename)
{
  cv::Mat pairwise_label_incompatibility_tmp;
  cv::Mat pairwise_kernel_weights_tmp;

  try
  {
    cv::FileStorage crf_parameter_file(input_parameter_filename, cv::FileStorage::READ);
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
    THEA_ERROR << e__.what();
    return false;
  }
  pairwise_label_incompatibility = pairwise_label_incompatibility_tmp.clone();
  pairwise_kernel_weights = pairwise_kernel_weights_tmp.clone();
  THEA_CONSOLE << "Imported trained CRF parameters.";
  THEA_CONSOLE << "Imported Label Compatibility Parameters: ";
  THEA_CONSOLE << pairwise_label_incompatibility;
  THEA_CONSOLE << "Imported Kernel weights: ";
  THEA_CONSOLE << pairwise_kernel_weights;

  return true;
}


bool MeshCRF::outputCRFParameters(const string& output_parameter_filename)
{
  try
  {
    cv::FileStorage crf_parameter_file(output_parameter_filename, cv::FileStorage::WRITE);
    crf_parameter_file << "pairwise_label_incompatibility" << pairwise_label_incompatibility;
    crf_parameter_file << "pairwise_kernel_weights" << pairwise_kernel_weights;
    crf_parameter_file.release();
  }
  catch (cv::Exception& e__)
  {
    THEA_ERROR << e__.what();
    return false;
  }
  return true;
}


float MeshCRF::mfinference(const std::shared_ptr<MeshProcessor>& mesh_processor_ptr, int max_iter, bool used_for_learning)
{
  // initialize mf and related probabilities
  int num_threads_to_use = omp_get_max_threads();
  omp_set_num_threads(num_threads_to_use);
  THEA_CONSOLE << "Initiating mean-field inference for mesh " << mesh_processor_ptr->getMeshPath() << "... (will use #threads=" << num_threads_to_use << ")";
  if (!mesh_processor_ptr->face_unary_probabilities.data || !mesh_processor_ptr->face_log_unary_features.data) // skip fcn mode, assume data are stored in the right place
  {
    string input_filename = Thea::FilePath::concat(crf_input_output_folder, Thea::FilePath::baseName(mesh_processor_ptr->getMeshPath()) + "_crf_unary_features.txt");
    THEA_CONSOLE << "Attempting to import unary term probabilities from " << input_filename;
    if (!mesh_processor_ptr->inputCRFUnaryFeatures(input_filename))
    {
      THEA_ERROR << "Cannot perform mean field since CRF unary term probabilities are not available!!!";
      return 0.0f;
    }
  }
  if (mesh_processor_ptr->face_pairwise_features_flattened.empty())
  {
    string input_filename = Thea::FilePath::concat(crf_input_output_folder, Thea::FilePath::baseName(mesh_processor_ptr->getMeshPath()) + "_crf_pairwise_features.bin");
    THEA_CONSOLE << "Attempting to import pairwise features from " << input_filename;
    if (!mesh_processor_ptr->inputCRFPairwiseFeatures(input_filename))
    {
      THEA_ERROR << "Cannot perform mean field since CRF pairwise features are not available!!!";
      return 0.0f;
    }
  }

  cv::Mat logQ = mesh_processor_ptr->face_log_unary_features.clone();
  cv::Mat row_sum(mesh_processor_ptr->number_of_faces, 1, CV_32F);
  cv::Mat row_max(mesh_processor_ptr->number_of_faces, 1, CV_32F);
  mesh_processor_ptr->face_mf_probabilities = mesh_processor_ptr->face_unary_probabilities.clone(); // initialize mean-field with unary term
  cv::Mat old_face_mf_probabilities = mesh_processor_ptr->face_mf_probabilities.clone();

  float accuracy = 0.0f;
  if (!used_for_learning)
  {    
    accuracy = mesh_processor_ptr->computeMeshLabelingAccuracy(label_map);
    THEA_LOG << "Initial mesh labeling accuracy: " << 100.0f * accuracy;
  }  
  
  ///// for debugging [check what happens at the initilization state]
  //mesh_processor_ptr->outputMFlabels("tmp.txt", label_map);
  //system("pause");
  
  // save all nodes so that we use openmp (<3.0v)
  vector< vector< std::tuple<int, int, float> > > nodes(mesh_processor_ptr->number_of_faces);
  for (unsigned long fpe = 0; fpe < mesh_processor_ptr->face_pairwise_features_flattened.size(); fpe += 3)
  {
    int feature_id = (int)mesh_processor_ptr->face_pairwise_features_flattened[fpe] / Settings::max_number_of_faces;
    int face_id1 = (int)mesh_processor_ptr->face_pairwise_features_flattened[fpe] % Settings::max_number_of_faces;
    int face_id2 = (int)mesh_processor_ptr->face_pairwise_features_flattened[fpe + 1];
    float dissimilarity = mesh_processor_ptr->face_pairwise_features_flattened[fpe + 2];
    nodes[face_id1].push_back(std::make_tuple(feature_id, face_id2, dissimilarity));
  }

  //// MAYBE LABEL WEIGHTS ARE IMPORTANT - TO CHECK LATER
  //vector<float> label_weights(label_map.size(), 1e-5f);
  //if (used_for_learning)
  //{
  //  for (int f = 0; f < mesh_processor_ptr->ground_truth_face_labels.size(); f++)
  //    label_weights[mesh_processor_ptr->ground_truth_face_labels[f]]++;

  //  for (int l = 0; l < label_map.size(); l++)
  //  {
  //    label_weights[l] = (float)mesh_processor_ptr->number_of_faces / label_weights[l];
  //    std::cout << label_weights[l] << std::endl;
  //  }
  //}



  // MF iterations (must be fast!)
  for (int mf_iter = 1; mf_iter <= max_iter; ++mf_iter)
  {
    old_face_mf_probabilities = mesh_processor_ptr->face_mf_probabilities.clone(); 
    logQ = mesh_processor_ptr->face_log_unary_features.clone();

#pragma omp parallel for
    for (int node = 0; node < (int)mesh_processor_ptr->number_of_faces; ++node)
    {
      for (unsigned int adj_node = 0; adj_node < nodes[node].size(); ++adj_node)
      {
        std::tuple<int, int, float>& entry = nodes[node][adj_node];
        float dissimilarity = std::get<2>(entry);
        float kernel_weight = pairwise_kernel_weights.at<float>(std::get<0>(entry));

        for (int c2 = 0; c2 < label_map.size(); ++c2)
        {
          float nbr_mf_prob = mesh_processor_ptr->face_mf_probabilities.at<float>(std::get<1>(entry), c2);
          float nbr_mf_prob_mult_kernel = nbr_mf_prob * kernel_weight;

          for (int c = 0; c < label_map.size(); ++c)
          {
            if (c != c2)
            {
              logQ.at<float>(node, c) -= nbr_mf_prob_mult_kernel * pairwise_label_incompatibility.at<float>(c, c2) * (1.0f - dissimilarity);
              //logQ.at<float>(node, c) -= nbr_mf_prob_mult_kernel * pairwise_label_incompatibility.at<float>(c, c2) * exp(-dissimilarity); // more like previous dense crfs
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
    cv::exp(logQ, mesh_processor_ptr->face_mf_probabilities);
    cv::reduce(mesh_processor_ptr->face_mf_probabilities, row_sum, 1, CV_REDUCE_SUM);
    for (int i = 0; i < mesh_processor_ptr->face_mf_probabilities.rows; ++i)
      mesh_processor_ptr->face_mf_probabilities.row(i) /= row_sum.at<float>(i);
    
    float diff = cv::sum(cv::abs(old_face_mf_probabilities - mesh_processor_ptr->face_mf_probabilities))[0] / float(mesh_processor_ptr->number_of_faces);
    if (diff < 1e-5f)
    {
      THEA_CONSOLE << "Converged.";
      break;
    }
      
    if (!used_for_learning)
    {
      accuracy = mesh_processor_ptr->computeMeshLabelingAccuracy(label_map);
      THEA_CONSOLE << "Iteration " << mf_iter  << ", mesh labeling accuracy : " << 100.0f * accuracy << ", avg change in marginals: " << diff;
    }

    ///// for debugging [check what happens at each iteration]
    //mesh_processor_ptr->computeMeshLabelingAccuracy(label_map);
    //mesh_processor_ptr->outputMFlabels("tmp.txt", label_map);
    //system("pause");
  }

  accuracy = mesh_processor_ptr->computeMeshLabelingAccuracy(label_map);
  THEA_LOG << "=> Mesh accuracy for mesh: " << mesh_processor_ptr->getMeshPath() << " [accuracy = " << 100.0f * accuracy << "]";

  if (!used_for_learning)
  {
    if (called_from_joint_mvfcn_model)
    {
      mesh_processor_ptr->outputMFprobs(Thea::FilePath::concat(crf_input_output_folder, Thea::FilePath::baseName(mesh_processor_ptr->getMeshPath()) + "_crf_prob_output.txt"));
      mesh_processor_ptr->outputMFlabels(Thea::FilePath::concat(crf_input_output_folder, Thea::FilePath::baseName(mesh_processor_ptr->getMeshPath()) + "_crf_label_output.txt"), label_map);
    }
    else
    {
      mesh_processor_ptr->outputMFprobs(Thea::FilePath::concat(crf_input_output_folder, Thea::FilePath::baseName(mesh_processor_ptr->getMeshPath()) + "_disjoint_crf_prob_output.txt"));
      mesh_processor_ptr->outputMFlabels(Thea::FilePath::concat(crf_input_output_folder, Thea::FilePath::baseName(mesh_processor_ptr->getMeshPath()) + "_disjoint_crf_label_output.txt"), label_map);
    }
  } 
  else
  {
    //// some corrections might be impossible without 'smoothing' the probs
    mesh_processor_ptr->face_mf_probabilities += 1e-5f; // equivalent to 'robust' log-likelihood
    cv::reduce(mesh_processor_ptr->face_mf_probabilities, row_sum, 1, CV_REDUCE_SUM);
    for (int i = 0; i < mesh_processor_ptr->face_mf_probabilities.rows; ++i)
      mesh_processor_ptr->face_mf_probabilities.row(i) /= row_sum.at<float>(i);

    mesh_derivative_pairwise_label_incompatibility = cv::Mat::zeros((int)label_map.size(), (int)label_map.size(), CV_32F);
    mesh_derivative_pairwise_kernel_weights = cv::Mat::zeros(pairwise_kernel_weights.rows, 1, CV_32F);
    vector< cv::Mat > mesh_derivative_pairwise_label_incompatibility_threads(num_threads_to_use);
    vector< cv::Mat > mesh_derivative_pairwise_kernel_weights_threads(num_threads_to_use);
    for (int tid = 0; tid < num_threads_to_use; tid++)
    {
      mesh_derivative_pairwise_label_incompatibility_threads[tid] = cv::Mat::zeros((int)label_map.size(), (int)label_map.size(), CV_32F);
      mesh_derivative_pairwise_kernel_weights_threads[tid] = cv::Mat::zeros(pairwise_kernel_weights.rows, 1, CV_32F);
    }
      

#pragma omp parallel for
    for (int node = 0; node < (int)mesh_processor_ptr->number_of_faces; ++node)
    {
      for (unsigned int adj_node = 0; adj_node < nodes[node].size(); ++adj_node)
      {
        std::tuple<int, int, float>& entry = nodes[node][adj_node];
        int ground_truth_label_n = mesh_processor_ptr->ground_truth_face_labels[node];
        int ground_truth_label_nbr = mesh_processor_ptr->ground_truth_face_labels[std::get<1>(entry)];
        float dissimilarity = std::get<2>(entry);
        float kernel_weight = pairwise_kernel_weights.at<float>(std::get<0>(entry));
        int tid = omp_get_thread_num();

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
        for (int c2 = 0; c2 < label_map.size(); ++c2)
        {
          float nbr_mf_prob = mesh_processor_ptr->face_mf_probabilities.at<float>(std::get<1>(entry), c2);

          for (int c = 0; c < label_map.size(); ++c)
          {
            float n_mf_prob = mesh_processor_ptr->face_mf_probabilities.at<float>(std::get<1>(entry), c);

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
    float counts = (float)mesh_processor_ptr->getNumberOfEntriesInPairwiseFeatures() / 3.0f;
    mesh_derivative_pairwise_label_incompatibility = mesh_derivative_pairwise_label_incompatibility / counts;
    mesh_derivative_pairwise_kernel_weights = mesh_derivative_pairwise_kernel_weights / counts;
  } // end of if condition for learning

  return accuracy;
}

#endif
