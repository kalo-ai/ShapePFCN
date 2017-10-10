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

#include "caffe/layers/mesh_image_label_data_layer.hpp"


namespace caffe {

template <typename Dtype>
MeshImageLabelDataLayer<Dtype>::MeshImageLabelDataLayer(
    const LayerParameter &param) : BasePrefetchingDataLayer<Dtype>(param, true, true, true) {
  std::random_device rand_dev;
  rng_ = new std::mt19937(rand_dev());
  max_num_views_ = -1;
  max_num_faces_ = -1;
  max_num_pairwise_entries_ = -1;
  num_pairwise_features_ = -1;  
  batch_size_ = -1;
  stochastic_ = false;
  image_margin_ = -1;
  mesh_id_ = 0;
  crop_size_ = -1;
  crop_minus_margin_size_ = -1;    
  image_mean_ = 0.0f;
  depth_mean_ = 0.0f;
  aux_mean_ = 0.0f; // SDF/UP change
  data_shape.resize(4);
  image2mesh_data_shape.resize(4);
  crfpairwise_data_shape.resize(4);
  label_shape.resize(4);
}

template <typename Dtype>
MeshImageLabelDataLayer<Dtype>::~MeshImageLabelDataLayer() {
  this->StopInternalThread();
  delete rng_;
}

template <typename Dtype>
void MeshImageLabelDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) 
{
  // the function setups the following top data:
  // top[0] => image RGB + depth + aux (sdf): [batch size] x [#views * 3] x [crop_size] x [crop_size] => this will be split from a slice layer
  // top[1] => triangle IDs (1 channel encoding triangle ids): [batch size] x [#views] x [crop_minus_margin_size] x [crop_minus_margin_size] => this will go as-is to the image2mesh layer
  // top[2] => pairwise CRF features: [batch size] x 1 x MAX_NUMBER_PAIRWISE_ENTRIES] x 1
  // top[3] => mesh labels: [batch size] x 1 x [max_num_faces] x 1  => this will go as-is to the image2mesh/loss/accuracy layers
  
  const MeshImageLabelDataParameter &data_param = this->layer_param_.mesh_image_label_data_param();
  mesh_list_filename_ = data_param.mesh_list_filename();
  rendered_image_dir_ = data_param.rendered_image_dir() + "/";
  depth_image_dir_ = data_param.depth_image_dir() + "/"; 
  aux_image_dir_ = data_param.aux_image_dir() + "/"; // SDF/UP change
  rendered_triangleid_dir_ = data_param.rendered_triangleid_dir() + "/";
  crf_features_dir_ = data_param.crf_features_dir() + "/";
  std::replace(rendered_image_dir_.begin(), rendered_image_dir_.end(), '\\', '/');
  std::replace(depth_image_dir_.begin(), depth_image_dir_.end(), '\\', '/');
  std::replace(aux_image_dir_.begin(), aux_image_dir_.end(), '\\', '/'); // SDF/UP change
  std::replace(rendered_triangleid_dir_.begin(), rendered_triangleid_dir_.end(), '\\', '/');
  std::replace(crf_features_dir_.begin(), crf_features_dir_.end(), '\\', '/');

  max_num_views_ = data_param.max_num_views();
  max_num_faces_ = data_param.max_num_faces();
  max_num_pairwise_entries_ = data_param.max_num_pairwise_entries();
  num_pairwise_features_ = data_param.num_pairwise_features();  
  batch_size_ = data_param.batch_size();
  stochastic_ = data_param.stochastic();
  image_margin_ = data_param.image_margin();
  image_mean_ = data_param.image_mean();
  depth_mean_ = data_param.depth_mean();
  aux_mean_ = data_param.aux_mean(); // SDF/UP change

  CHECK_GT(max_num_views_, 0) << "Positive #views required";
  CHECK_GT(max_num_faces_, 0) << "Positive #faces required";
  CHECK_GE(max_num_pairwise_entries_, 0) << "Positive or 0 #max pairwise entries required";
  CHECK_GE(num_pairwise_features_, 0) << "Positive or 0 #pairwise features required";
  CHECK_GT(image_margin_, 0) << "Positive image size required";
  CHECK_GT(batch_size_, 0) << "Positive batch size required";
  
  // Read the file with mesh list
  LOG(INFO) << "Opening mesh list " << mesh_list_filename_;
  mesh_filename_lines_.clear();
  std::ifstream infile(mesh_list_filename_.c_str());
  string filename;
  while (infile >> filename) 
    mesh_filename_lines_.push_back(filename);
  infile.close();

  // shuffle the list
  if (data_param.shuffle()) 
  {
    // randomly shuffle data
    LOG(INFO) << "Shuffling mesh list";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleMeshes();
  }
  LOG(INFO) << "There is a total of " << mesh_filename_lines_.size() << " meshes.";

  // try to load a labels file
  string base_mesh_filename = mesh_filename_lines_[0];
  std::replace(base_mesh_filename.begin(), base_mesh_filename.end(), '\\', '/'); 
  size_t last_index = base_mesh_filename.find_last_of(".");
  if (last_index != string::npos)
    base_mesh_filename = base_mesh_filename.substr(0, last_index);  
  string label_filename = base_mesh_filename + ".seg";
  std::ifstream label_file(label_filename.c_str());
  if (label_file.good())
    LOG(INFO) << "At least one labels file found (necessary for training)";
  else
    LOG(INFO) << "Labels file could not be found - labeling accuracy cannot be evaluated (note: this is ok during testing on novel data)";
  label_file.close();

  // try to load a rendered image and corresponding depth + triangle id image
  last_index = base_mesh_filename.find_last_of("/");
  if (last_index != string::npos)
    base_mesh_filename = base_mesh_filename.substr(last_index+1);
  vector<string> rendered_image_filenames = searchForImagesOfAMesh(rendered_image_dir_, base_mesh_filename);
  LOG(INFO) << "Preliminary check: there is a total of " << rendered_image_filenames.size() << " views for mesh " << mesh_filename_lines_[0];
  CHECK_GT(rendered_image_filenames.size(), 0);
  string depth_image_filename = rendered_image_filenames[0];
  boost::replace_all(depth_image_filename, base_mesh_filename + "_int_", base_mesh_filename + "_dep_");
  boost::replace_all(depth_image_filename, rendered_image_dir_, depth_image_dir_);
  string aux_image_filename = rendered_image_filenames[0]; // SDF/UP change
  boost::replace_all(aux_image_filename, base_mesh_filename + "_int_", base_mesh_filename + "_aux_"); // SDF/UP change
  boost::replace_all(aux_image_filename, rendered_image_dir_, aux_image_dir_); // SDF/UP change
  string triangleid_image_filename = rendered_image_filenames[0];
  boost::replace_all(triangleid_image_filename, base_mesh_filename + "_int_", base_mesh_filename + "_fid_");
  boost::replace_all(triangleid_image_filename, rendered_image_dir_, rendered_triangleid_dir_);
  cv::Mat temp_cv_img = ReadImageToCVMat(rendered_image_filenames[0], false);
  CHECK(temp_cv_img.data) << "Could not load " << rendered_image_filenames[0];
  cv::Mat temp_cv_depth_img = ReadImageToCVMat(depth_image_filename, false);
  CHECK(temp_cv_depth_img.data) << "Could not load depth image: " << depth_image_filename;
  cv::Mat temp_cv_aux_img = ReadImageToCVMat(aux_image_filename, false); // SDF/UP change
  CHECK(temp_cv_aux_img.data) << "Could not load aux image: " << aux_image_filename; // SDF/UP change
  cv::Mat temp_cv_tid_img = ReadImageToCVMat(triangleid_image_filename, true);
  CHECK(temp_cv_tid_img.data) << "Could not load triangle id image: " << triangleid_image_filename;


  // get crop size from layer parameters
  crop_size_ = temp_cv_img.rows;
  const TransformationParameter& transform_param = this->layer_param_.transform_param();
  if (transform_param.has_crop_size())
    crop_size_ = transform_param.crop_size();
  LOG(INFO) << "Will use crop size " << crop_size_;
  crop_minus_margin_size_ = crop_size_ - 2 * image_margin_;
  CHECK_GT(crop_size_, 0) << "Crop size should have been specified as input parameter (and should have been positive)";
  CHECK_GT(crop_minus_margin_size_, 0) << "Crop size should have been bigger than 2 * margin size";

  // try to load pairwise features
  if (max_num_pairwise_entries_ > 0 && num_pairwise_features_ > 0)
  {
    string crf_features_filename = crf_features_dir_ + base_mesh_filename + "_crf_pairwise_features.bin";
    std::ifstream crf_feature_file;
    crf_feature_file.open(crf_features_filename, ios::in | ios::binary);
    CHECK(crf_feature_file.good()) << "Could not load " << crf_features_filename;;
    crf_feature_file.close();
  }
  else
  {
    LOG(INFO) << "max_num_pairwise_entries is set to 0 - will not load pairwise features, pairwise data blob will be empty (crf loss should not be used in this case)";
  }

  // determine data blob shape
  data_shape[0] = batch_size_;
  data_shape[1] = 3 * max_num_views_;                // will concatenate all mesh images across views // 2=>3 SDF change
  data_shape[2] = crop_size_;
  data_shape[3] = crop_size_;
  top[0]->Reshape(data_shape);

  // determine triangleid blob shape
  image2mesh_data_shape[0] = batch_size_;
  image2mesh_data_shape[1] = max_num_views_;                // will concatenate all mesh images across views - 1 channel storing face ids directly  
  image2mesh_data_shape[2] = crop_minus_margin_size_;   // after passing through the convnet, each cropped image will lose border areas
  image2mesh_data_shape[3] = crop_minus_margin_size_;
  top[1]->Reshape(image2mesh_data_shape); // triangleid data use same dimensions with rendered data

  // determine pairwise crf features blob shape
  if (max_num_pairwise_entries_ > 0 && num_pairwise_features_ > 0)
  {
    crfpairwise_data_shape[0] = batch_size_;
    crfpairwise_data_shape[1] = 1;
    crfpairwise_data_shape[2] = max_num_pairwise_entries_;
    crfpairwise_data_shape[3] = 1;
  }
  else
  {
    crfpairwise_data_shape.resize(0);  // =scalar; 
  }
  top[2]->Reshape(crfpairwise_data_shape); // triangleid data use same dimensions with rendered data

  // determine label blob shape
  label_shape[0] = batch_size_;
  label_shape[1] = 1;
  label_shape[2] = max_num_faces_;  // last entry is used to save #faces per mesh
  label_shape[3] = 1;
  top[3]->Reshape(label_shape);

  // prepare prefetch data
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) 
  {
    this->prefetch_[i].data_.Reshape(data_shape);
    this->prefetch_[i].image2mesh_data_.Reshape(image2mesh_data_shape);
    this->prefetch_[i].crfpairwise_data_.Reshape(crfpairwise_data_shape);
    this->prefetch_[i].label_.Reshape(label_shape);
  }

  // output info
  LOG(INFO) << "output data size: " << top[0]->num() << ","
  << top[0]->channels() << "," << top[0]->height() << ","
  << top[0]->width();

  LOG(INFO) << "output image2meshdata size: " << top[1]->num() << ","
    << top[1]->channels() << "," << top[1]->height() << ","
    << top[1]->width();

  LOG(INFO) << "output crfpairwisedata size: " << top[2]->num() << ","
    << top[2]->channels() << "," << top[2]->height() << ","
    << top[2]->width();

  LOG(INFO) << "output label size: " << top[3]->num() << ","
    << top[3]->channels() << "," << top[3]->height() << ","
    << top[3]->width();
}



template <typename Dtype>
void MeshImageLabelDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  // setup timer 
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0.0;
  double trans_time = 0.0;
  double preprocess_time = 0.0;
  CPUTimer timer;
  timer.Start();

  // layer params double check
  CHECK_GT(max_num_views_, 0) << "Positive #views required";
  CHECK_GT(max_num_faces_, 0) << "Positive #faces required";
  CHECK_GE(max_num_pairwise_entries_, 0) << "Positive or 0 #max pairwise entries required";
  CHECK_GE(num_pairwise_features_, 0) << "Positive or 0 #pairwise features required";
  CHECK_GT(image_margin_, 0) << "Positive image size required";
  CHECK_GT(batch_size_, 0) << "Positive batch size required";
  CHECK_GT(crop_size_, 0) << "Crop size should have been specified as input parameter (and should have been positive)";
  CHECK_GT(crop_minus_margin_size_, 0) << "Crop size should have been bigger than 2 * margin size";
  CHECK_GT(mesh_filename_lines_.size(), 0) << "No training/test mesh filenames specified";

  if (stochastic_)
  {
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  }

  // setup batch blob shapes
  batch->data_.Reshape(data_shape);
  batch->image2mesh_data_.Reshape(image2mesh_data_shape);
  batch->crfpairwise_data_.Reshape(crfpairwise_data_shape);
  batch->label_.Reshape(label_shape);

  // debug info
  DLOG(INFO) << "batch data size: " << batch->data_.num() << ","
    << batch->data_.channels() << "," << batch->data_.height() << ","
    << batch->data_.width();

  DLOG(INFO) << "batch image2meshdata size: " << batch->image2mesh_data_.num() << ","
    << batch->image2mesh_data_.channels() << "," << batch->image2mesh_data_.height() << ","
    << batch->image2mesh_data_.width();

  DLOG(INFO) << "batch crfpairwisedata size: " << batch->crfpairwise_data_.num() << ","
    << batch->crfpairwise_data_.channels() << "," << batch->crfpairwise_data_.height() << ","
    << batch->crfpairwise_data_.width();

  DLOG(INFO) << "batch label size: " << batch->label_.num() << ","
    << batch->label_.channels() << "," << batch->label_.height() << ","
    << batch->label_.width();

  // load all data into the batch
  size_t num_meshes = mesh_filename_lines_.size();
  preprocess_time = timer.MicroSeconds();
  for (int item_id = 0; item_id < batch_size_; ++item_id) 
  {
    timer.Start();
    CHECK_GT(num_meshes, mesh_id_);

    // load labels
    string base_mesh_filename = mesh_filename_lines_[mesh_id_];
    std::replace(base_mesh_filename.begin(), base_mesh_filename.end(), '\\', '/');
    size_t last_index = base_mesh_filename.find_last_of(".");
    if (last_index != string::npos)
      base_mesh_filename = base_mesh_filename.substr(0, last_index);
    string label_filename = base_mesh_filename + ".seg";
    std::ifstream label_file(label_filename.c_str());
    std::vector<int> face_labels;
    if (this->data_transformer_->getPhase() == TRAIN)
    {
      CHECK(label_file.good()) << "Could not load " << label_filename;
      face_labels = std::vector <int> { std::istream_iterator<int>(label_file), std::istream_iterator<int>() };
      DLOG(INFO) << "read " << face_labels.size() << " face labels from " << label_filename;
      CHECK_GE(max_num_faces_, face_labels.size()) << "#faces > max supported #faces";
    }
    else
    {
      if (label_file.good())
      {
        face_labels = std::vector <int> { std::istream_iterator<int>(label_file), std::istream_iterator<int>() };
        DLOG(INFO) << "read " << face_labels.size() << " from " << label_filename;
        CHECK_GE(max_num_faces_, face_labels.size()) << "#faces > max supported #faces";
      }
      else
      {
        LOG(INFO) << "Labels file could not be found - labeling accuracy cannot be evaluated (note: this is ok during testing on novel data)";
      }
    }
    label_file.close();

    // load image
    last_index = base_mesh_filename.find_last_of("/");
    if (last_index != string::npos)
      base_mesh_filename = base_mesh_filename.substr(last_index + 1);
    vector<string> rendered_image_filenames = searchForImagesOfAMesh(rendered_image_dir_, base_mesh_filename);
    DLOG(INFO) << "There is a total of " << rendered_image_filenames.size() << " views for mesh in the batch: " << mesh_filename_lines_[mesh_id_];
    if (stochastic_)
    {
      ShuffleImagesOfMesh(rendered_image_filenames);
    }
    CHECK_GT(rendered_image_filenames.size(), 0);
    vector<cv::Mat> all_views_cv_img_channels;
    vector<cv::Mat> all_views_cv_tid_img_channels;

    // load pairwise features
    std::vector<float> mesh_crfpairwise_data;
    if (max_num_pairwise_entries_ > 0 && num_pairwise_features_ > 0)
    {
      string crf_features_filename = crf_features_dir_ + base_mesh_filename + "_crf_pairwise_features.bin";
      std::ifstream crf_feature_file;
      crf_feature_file.open(crf_features_filename, ios::in | ios::binary);
      CHECK(crf_feature_file.good()) << "Could not load " << crf_features_filename;
      crf_feature_file.seekg(0, ios::end);
      long number_of_entries_in_pairwise_features = crf_feature_file.tellg() / sizeof(float);
      CHECK_GE( (long)max_num_pairwise_entries_, number_of_entries_in_pairwise_features) << "#pairwise entries > max supported #pairwise entries";
      crf_feature_file.seekg(0, ios::beg);
      mesh_crfpairwise_data.resize(number_of_entries_in_pairwise_features);
      crf_feature_file.read(reinterpret_cast<char*>(&mesh_crfpairwise_data[0]), number_of_entries_in_pairwise_features*sizeof(float));
      crf_feature_file.close();
      CHECK_GE(max_num_pairwise_entries_, mesh_crfpairwise_data.size()) << "#pairwise entries > max supported #pairwise entries";
    }
    
    for (int view_id = 0; view_id < max_num_views_; view_id++)
    {
      // read image for view
      string rendered_image_filename = rendered_image_filenames[view_id % rendered_image_filenames.size()];
      string depth_image_filename = rendered_image_filename;
      boost::replace_all(depth_image_filename, base_mesh_filename + "_int_", base_mesh_filename + "_dep_");
      boost::replace_all(depth_image_filename, rendered_image_dir_, depth_image_dir_);
      string aux_image_filename = rendered_image_filename; // SDF/UP change
      boost::replace_all(aux_image_filename, base_mesh_filename + "_int_", base_mesh_filename + "_aux_"); // SDF/UP change
      boost::replace_all(aux_image_filename, rendered_image_dir_, aux_image_dir_); // SDF/UP change
      string triangleid_image_filename = rendered_image_filename;
      boost::replace_all(triangleid_image_filename, base_mesh_filename + "_int_", base_mesh_filename + "_fid_");
      boost::replace_all(triangleid_image_filename, rendered_image_dir_, rendered_triangleid_dir_);
      cv::Mat cv_img = ReadImageToCVMat(rendered_image_filename, false);
      cv::Mat cv_depth_img = ReadImageToCVMat(depth_image_filename, false);
      cv::Mat cv_aux_img = ReadImageToCVMat(aux_image_filename, false); // SDF/UP change
      cv::Mat cv_tid_img = ReadImageToCVMat(triangleid_image_filename, true);
      CHECK(cv_img.data) << "Could not load rendered image: " << rendered_image_filename;
      CHECK(cv_depth_img.data) << "Could not load depth image: " << depth_image_filename;
      CHECK(cv_aux_img.data) << "Could not load aux image: " << aux_image_filename; // SDF/UP change
      CHECK(cv_tid_img.data) << "Could not load triangle id image: " << triangleid_image_filename;
      DLOG(INFO) << "Read: " << rendered_image_filename << " (mesh " << mesh_filename_lines_[mesh_id_] << ", item_id=" << item_id << ", mesh_id=" << mesh_id_ << ", phase: " << this->data_transformer_->getPhase() << ")";
      DLOG(INFO) << "Read: " << depth_image_filename << " (mesh " << mesh_filename_lines_[mesh_id_] << ", item_id=" << item_id << ", mesh_id=" << mesh_id_ << ", phase: " << this->data_transformer_->getPhase() << ")";
      DLOG(INFO) << "Read: " << aux_image_filename << " (mesh " << mesh_filename_lines_[mesh_id_] << ", item_id=" << item_id << ", mesh_id=" << mesh_id_ << ", phase: " << this->data_transformer_->getPhase() << ")"; // SDF/UP change
      DLOG(INFO) << "Read: " << triangleid_image_filename << " (mesh " << mesh_filename_lines_[mesh_id_] << ", item_id=" << item_id << ", mesh_id=" << mesh_id_ << ", phase: " << this->data_transformer_->getPhase() << ")";
      vector<cv::Mat> cv_transformed_img;
      cv::Mat cv_transformed_tid_img;

      switch (this->layer_param_.mesh_image_label_data_param().padding())
      {
      case MeshImageLabelDataParameter_Padding_ZERO:
        cv_img = ExtendLabelMargin(cv_img, image_margin_, image_margin_, 0);
        cv_depth_img = ExtendLabelMargin(cv_depth_img, image_margin_, image_margin_, 0);
        cv_aux_img = ExtendLabelMargin(cv_aux_img, image_margin_, image_margin_, 0); // SDF/UP change
        cv_tid_img = ExtendLabelMargin(cv_tid_img, image_margin_, image_margin_, 0);
        break;
      case MeshImageLabelDataParameter_Padding_REFLECT:
        cv_img = ExtendLabelMargin(cv_img, image_margin_, image_margin_, -1);
        cv_depth_img = ExtendLabelMargin(cv_depth_img, image_margin_, image_margin_, -1);
        cv_aux_img = ExtendLabelMargin(cv_aux_img, image_margin_, image_margin_, -1); // SDF/UP change
        cv_tid_img = ExtendLabelMargin(cv_tid_img, image_margin_, image_margin_, -1);
        break;
      default:
        LOG(FATAL) << "Unknown Padding";
      }
      DLOG(INFO) << "Rendered image size: " << cv_img.size() << ", channels: " << cv_img.channels();
      DLOG(INFO) << "Depth image size: " << cv_depth_img.size() << ", channels: " << cv_depth_img.channels();
      DLOG(INFO) << "Aux image size: " << cv_aux_img.size() << ", channels: " << cv_aux_img.channels(); // SDF/UP change
      DLOG(INFO) << "Triangle id image size: " << cv_tid_img.size() << ", channels: " << cv_tid_img.channels();
      read_time += timer.MicroSeconds();
      timer.Start();

      // apply transformations and concatenate all views into a single big image
      Transform(cv_img, cv_depth_img, cv_aux_img, cv_tid_img, cv_transformed_img, cv_transformed_tid_img); // SDF/UP change
      all_views_cv_img_channels.insert(all_views_cv_img_channels.end(), cv_transformed_img.begin(), cv_transformed_img.end());
      all_views_cv_tid_img_channels.push_back(cv_transformed_tid_img); // 1 channel after transformation
    } // end of loop over views
    DLOG(INFO) << "Concatenated total " << all_views_cv_img_channels.size() << " channels of rendered data.";
    DLOG(INFO) << "Concatenated total " << all_views_cv_tid_img_channels.size() << " channels of triangleid data.";
   
    // rendered data => batch
    int mem_index = 0;
    int rendered_image_offset = batch->data_.offset(item_id);
    Dtype* transformed_data = batch->data_.mutable_cpu_data() + rendered_image_offset;
    for (int h = 0; h < crop_size_; ++h)
    {
      for (int w = 0; w < crop_size_; ++w)
      {
        for (int c = 0; c < max_num_views_ * 3; ++c)  // 2=>3 SDF change
        {
          mem_index = (c * crop_size_ + h) * crop_size_ + w;
          Dtype pixel_value = static_cast<Dtype>(all_views_cv_img_channels[c].at<float>(h, w));
          transformed_data[mem_index] = pixel_value;
        }
      }
    }    

    int triangleid_image_offset = batch->image2mesh_data_.offset(item_id);
    Dtype* transformed_image2mesh_data = batch->image2mesh_data_.mutable_cpu_data() + triangleid_image_offset;
    // triangleid data => batch
    for (int h = 0; h < crop_minus_margin_size_; ++h)
    {
      for (int w = 0; w < crop_minus_margin_size_; ++w)
      {
        for (int c = 0; c < max_num_views_; ++c)
        {
          mem_index = (c * crop_minus_margin_size_ + h) * crop_minus_margin_size_ + w;
          Dtype pixel_value = static_cast<Dtype>(all_views_cv_tid_img_channels[c].at<float>(h, w));
          transformed_image2mesh_data[mem_index] = pixel_value;
        }
      }
    }


    // crfpairwise features => batch
    if (max_num_pairwise_entries_ > 0 && num_pairwise_features_ > 0)
    {
      int crfpairwise_data_offset = batch->crfpairwise_data_.offset(item_id);
      Dtype *crfpairwise_data = batch->crfpairwise_data_.mutable_cpu_data() + crfpairwise_data_offset;
      for (int e = 0; e < max_num_pairwise_entries_; ++e)
      {
        if ( e >= mesh_crfpairwise_data.size() )
          crfpairwise_data[e] = static_cast<Dtype>(-1.0f);
        else
          crfpairwise_data[e] = static_cast<Dtype>(mesh_crfpairwise_data[e]);
      }
    }
    else
    {
      batch->crfpairwise_data_.mutable_cpu_data()[0] = Dtype(0);
    }

    // labels => batch
    int label_offset = batch->label_.offset(item_id);
    Dtype *label_data = batch->label_.mutable_cpu_data() + label_offset;
    for (int f = 0; f < max_num_faces_; ++f)
    {
      if (f >= face_labels.size() )
        label_data[f] = static_cast<Dtype>(255);
      else
        if (face_labels[f] < 0)
          label_data[f] = static_cast<Dtype>(255);
        else
          label_data[f] = static_cast<Dtype>( face_labels[f] );
    }
    
    trans_time += timer.MicroSeconds();

    // go to the next iter
    mesh_id_++;
    if (mesh_id_ >= num_meshes) 
    {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      mesh_id_ = 0;
      if (this->layer_param_.mesh_image_label_data_param().shuffle())
        ShuffleMeshes();
    }
  }

  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "Preprocess time: " << preprocess_time / 1000 << " ms.";
  DLOG(INFO) << "      Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << " Transform time: " << trans_time / 1000 << " ms.";
} 
 
template <typename Dtype>
cv::Mat MeshImageLabelDataLayer<Dtype>::ExtendLabelMargin(cv::Mat &image, int margin_w, int margin_h, double value)
{
  cv::Mat big_image;
  if (value < 0) {
    cv::copyMakeBorder(image, big_image, margin_h, margin_h, margin_w, margin_w, cv::BORDER_REFLECT_101);
  }
  else {
    cv::copyMakeBorder(image, big_image, margin_h, margin_h, margin_w, margin_w, cv::BORDER_CONSTANT, cv::Scalar(value));
  }
  return big_image;
}


template <typename Dtype>
vector<string> MeshImageLabelDataLayer<Dtype>::searchForImagesOfAMesh(const string& search_path, const string &base_mesh_filename)
{
  CHECK(boost::filesystem::exists(search_path)) << "Directory " << search_path << " does not exist.";
  const std::regex pattern(base_mesh_filename + "_.*\\.png");

  vector<string> matching_image_filenames;

  boost::filesystem::directory_iterator end_itr; // Default ctor yields past-the-end
  for (boost::filesystem::directory_iterator i(search_path); i != end_itr; ++i)
  {
    // Skip if not a file
    if (!boost::filesystem::is_regular_file(i->status())) continue;

    // Skip if no match:
    if (!std::regex_match(i->path().filename().string(), pattern)) continue;

    // File matches, store it
    matching_image_filenames.push_back(search_path + i->path().filename().string());
  }

  return matching_image_filenames;
}


template <typename Dtype>
void MeshImageLabelDataLayer<Dtype>::ShuffleMeshes()
{
  caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  vector<int> order(mesh_filename_lines_.size());
  for (int i = 0; i < order.size(); ++i)
    order[i] = i;

  shuffle(order.begin(), order.end(), prefetch_rng);
  vector<std::string> new_mesh_filename_lines_(mesh_filename_lines_.size());
  for (int i = 0; i < order.size(); ++i) {
    new_mesh_filename_lines_[i] = mesh_filename_lines_[order[i]];
  }
  swap(mesh_filename_lines_, new_mesh_filename_lines_);
}



template <typename Dtype>
void MeshImageLabelDataLayer<Dtype>::ShuffleImagesOfMesh(vector<string>& rendered_image_filenames)
{
  caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  vector<int> order(rendered_image_filenames.size());
  for (int i = 0; i < order.size(); ++i)
    order[i] = i;

  shuffle(order.begin(), order.end(), prefetch_rng);
  //int selected_rotation_id = ((*prefetch_rng)() % 4);

  vector<std::string> new_rendered_image_filenames;
  for (int i = 0; i < order.size(); ++i) 
  {
    string filename = rendered_image_filenames[order[i]];
    //std::size_t found_ = filename.find_last_of("_");
    //std::size_t founddot = filename.find_last_of(".");
    //if (found_ == string::npos || founddot == string::npos)
    //  new_rendered_image_filenames.push_back(filename);
    //else
    //{
    //  int rotation_id = stoi(filename.substr(found_ + 1, founddot));
    //  if (rotation_id != selected_rotation_id)
    //    continue; 
      new_rendered_image_filenames.push_back(filename);
    //}
    }

  swap(rendered_image_filenames, new_rendered_image_filenames);
}



// perhaps this could into data_transformer, but for now let's leave it here
template<typename Dtype>
void MeshImageLabelDataLayer<Dtype>::Transform(const cv::Mat& cv_img, const cv::Mat& cv_depth_img, const cv::Mat& cv_aux_img,  const cv::Mat& cv_tid_img, vector<cv::Mat>& cv_transformed_img, cv::Mat& cv_transformed_tid_img) // SDF/UP change
{
  // input checks
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;
  const bool do_mirror = this->layer_param_.transform_param().mirror() && this->data_transformer_->Rand(2);
  DLOG(INFO) << "mirror for this input image: " << do_mirror;

  CHECK(cv_img.depth() == CV_8U) << "Rendered image type must be unsigned byte";
  CHECK(img_channels == 1) << "Rendered image should be made out of 1 channel";
  CHECK_GE(img_height, crop_size_) << "Crop size should have been specified as input parameter, should have been positive, and height of all images >= crop_size)";
  CHECK_GE(img_width, crop_size_) << "Crop size should have been specified as input parameter, should have been positive, and height of all images >= crop_size)";

  CHECK_EQ(cv_depth_img.rows, img_height) << "Depth and rendered image should have the same height";
  CHECK_EQ(cv_depth_img.cols, img_width) << "Depth and rendered image should have the same height";
  CHECK(cv_depth_img.depth() == CV_8U) << "Depth image type must be unsigned byte";
  CHECK(cv_depth_img.channels() == 1) << "Depth image should be made out of 1 channel";

  CHECK_EQ(cv_aux_img.rows, img_height) << "Aux and rendered image should have the same height"; // SDF/UP change
  CHECK_EQ(cv_aux_img.cols, img_width) << "Aux and rendered image should have the same height"; // SDF/UP change
  CHECK(cv_aux_img.depth() == CV_8U) << "Aux image type must be unsigned byte"; // SDF/UP change
  CHECK(cv_aux_img.channels() == 1) << "Aux image should be made out of 1 channel"; // SDF/UP change

  CHECK_EQ(cv_tid_img.rows, img_height) << "Triangle ids and rendered image should have the same height";
  CHECK_EQ(cv_tid_img.cols, img_width) << "Triangle ids and rendered image should have the same height";
  CHECK(cv_tid_img.depth() == CV_8U) << "Triangle id data type must be unsigned byte";
  CHECK(cv_tid_img.channels() == 3) << "Triangle id data should be made out of 3 channels";

  // cropping
  int h_off = 0;
  int w_off = 0;
  cv::Mat cv_cropped_img = cv_img.clone();
  cv::Mat cv_cropped_depth_img = cv_depth_img.clone();
  cv::Mat cv_cropped_aux_img = cv_aux_img.clone(); // SDF/UP change
  cv::Mat cv_cropped_tid_img = cv_tid_img.clone();
  // We only do random crop when we do training.
  if (this->data_transformer_->getPhase() == TRAIN)
  {
    h_off = this->data_transformer_->Rand(img_height - crop_size_ + 1);
    w_off = this->data_transformer_->Rand(img_width - crop_size_ + 1);
  }
  else
  {
    h_off = (img_height - crop_size_) / 2;
    w_off = (img_width - crop_size_) / 2;
  }
  cv::Rect roi(w_off, h_off, crop_size_, crop_size_);
  cv_cropped_img = cv_img(roi);
  cv_cropped_depth_img = cv_depth_img(roi);
  cv_cropped_aux_img = cv_aux_img(roi); // SDF/UP change
  cv_cropped_tid_img = cv_tid_img(roi);
  
  CHECK(cv_cropped_img.data);
  CHECK(cv_cropped_depth_img.data);
  CHECK(cv_cropped_aux_img.data); // SDF/UP change
  CHECK(cv_cropped_tid_img.data);
  DLOG(INFO) << "Cropped image size: " << cv_cropped_img.size() << ", channels: " << cv_cropped_img.channels();
  DLOG(INFO) << "Cropped depth image size: " << cv_cropped_depth_img.size() << ", channels: " << cv_cropped_depth_img.channels();
  DLOG(INFO) << "Cropped aux image size: " << cv_cropped_aux_img.size() << ", channels: " << cv_cropped_aux_img.channels(); // SDF/UP change
  DLOG(INFO) << "Cropped triangle id image size: " << cv_cropped_tid_img.size() << ", channels: " << cv_cropped_tid_img.channels();
  DLOG(INFO) << "Used height offset: " << h_off << ", width offset: " << w_off;

  cv::Mat cv_mirrored_cropped_img = cv_cropped_img.clone();
  cv::Mat cv_mirrored_cropped_depth_img = cv_cropped_depth_img.clone();
  cv::Mat cv_mirrored_cropped_aux_img = cv_cropped_aux_img.clone(); // SDF/UP change
  cv::Mat cv_mirrored_cropped_tid_img = cv_cropped_tid_img.clone();
  if (do_mirror)
  {
    cv::flip(cv_cropped_img, cv_mirrored_cropped_img, 1);
    cv::flip(cv_cropped_depth_img, cv_mirrored_cropped_depth_img, 1);
    cv::flip(cv_cropped_aux_img, cv_mirrored_cropped_aux_img, 1); // SDF/UP change
    cv::flip(cv_cropped_tid_img, cv_mirrored_cropped_tid_img, 1);
  }

  CHECK(cv_mirrored_cropped_img.data);
  CHECK(cv_mirrored_cropped_depth_img.data);
  CHECK(cv_mirrored_cropped_aux_img.data); // SDF/UP change
  CHECK(cv_mirrored_cropped_tid_img.data);

  // [unsigned char, 1 channel => float conversions for image data and mean subtraction] x 3 (image+depth+aux) => final image
  cv_mirrored_cropped_img.convertTo(cv_mirrored_cropped_img, CV_32F);
  cv_mirrored_cropped_depth_img.convertTo(cv_mirrored_cropped_depth_img, CV_32F);
  cv_mirrored_cropped_aux_img.convertTo(cv_mirrored_cropped_aux_img, CV_32F); // SDF/UP change
  DLOG(INFO) << "Rendered Image pixel (0,0) before mean subtraction: " << cv_mirrored_cropped_img.at<float>(0, 0);
  DLOG(INFO) << "Rendered Depth pixel (0,0) before mean subtraction: " << cv_mirrored_cropped_depth_img.at<float>(0, 0);
  DLOG(INFO) << "Rendered Aux pixel (0,0) before mean subtraction: " << cv_mirrored_cropped_aux_img.at<float>(0, 0); // SDF/UP change
  cv_mirrored_cropped_img -= image_mean_;
  cv_mirrored_cropped_depth_img -= depth_mean_;
  cv_mirrored_cropped_aux_img -= aux_mean_;  // SDF/UP change
  DLOG(INFO) << "Rendered Image pixel (0,0) after mean subtraction: " << cv_mirrored_cropped_img.at<float>(0, 0);
  DLOG(INFO) << "Rendered Depth pixel (0,0) after mean subtraction: " << cv_mirrored_cropped_depth_img.at<float>(0, 0);
  DLOG(INFO) << "Rendered Aux pixel (0,0) after mean subtraction: " << cv_mirrored_cropped_aux_img.at<float>(0, 0); // SDF/UP change

  cv_transformed_img.push_back(cv_mirrored_cropped_img);
  cv_transformed_img.push_back(cv_mirrored_cropped_depth_img); // note: if you want to ignore depth, replicate with cv_mirrored_cropped_img here
  cv_transformed_img.push_back(cv_mirrored_cropped_aux_img); // note: if you want to ignore aux, replicate with cv_mirrored_cropped_img here // SDF/UP change

  // unsigned char, BGR 3 channels encoding triangle ids => triangle ids, 1 float channel 
  DLOG(INFO) << "Triangle ID Image pixel (image_margin,image_margin) before conversion/cropping with margin: " << cv_mirrored_cropped_tid_img.at<cv::Vec3b>(image_margin_, image_margin_);
  cv_transformed_tid_img = cv::Mat(crop_minus_margin_size_, crop_minus_margin_size_, CV_32F);
  cv_transformed_tid_img.setTo(-1.0f); // background pixels
  for (int i = 0; i < crop_minus_margin_size_; ++i)
  {
    for (int j = 0; j < crop_minus_margin_size_; ++j)
    {
      unsigned int b = (unsigned int)cv_mirrored_cropped_tid_img.at<cv::Vec3b>(i + image_margin_, j + image_margin_)[0];
      unsigned int g = (unsigned int)cv_mirrored_cropped_tid_img.at<cv::Vec3b>(i + image_margin_, j + image_margin_)[1];
      unsigned int r = (unsigned int)cv_mirrored_cropped_tid_img.at<cv::Vec3b>(i + image_margin_, j + image_margin_)[2];
      if (r == 255 && g == 255 && b == 255) // no pixel->triangle association
        continue;
      unsigned int face_index = r + 256 * g + 65536 * b;
      CHECK_GT(max_num_faces_, face_index);
      cv_transformed_tid_img.at<float>(i, j) = (float)face_index;
    }
  }
  DLOG(INFO) << "Triangle ID Image pixel (0,0) after conversion/cropping with margin (should be RED+256*GREEN+65536*BLUE of the previous one, or -1 after background): " << cv_transformed_tid_img.at<float>(0, 0);
}



INSTANTIATE_CLASS(MeshImageLabelDataLayer);
REGISTER_LAYER_CLASS(MeshImageLabelData);

}  // namespace caffe
