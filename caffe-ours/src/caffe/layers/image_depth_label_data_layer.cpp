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

#include "caffe/layers/image_depth_label_data_layer.hpp"


namespace caffe {

  template <typename Dtype>
  ImageDepthLabelDataLayer<Dtype>::ImageDepthLabelDataLayer(
    const LayerParameter &param) : BasePrefetchingDataLayer<Dtype>(param, true, false, false) {
    std::random_device rand_dev;
    rng_ = new std::mt19937(rand_dev());
    validation_mode_ = -1;
    batch_size_ = -1;
    image_margin_ = -1;
    image_id_ = 0;
    crop_size_ = -1;
    crop_minus_margin_size_ = -1;
    image_mean_ = 0.0f;
    depth_mean_ = 0.0f;
    aux_mean_ = 0.0f; // SDF/UP change
    data_shape.resize(4);
    label_shape.resize(4);
  }

  template <typename Dtype>
  ImageDepthLabelDataLayer<Dtype>::~ImageDepthLabelDataLayer() {
    this->StopInternalThread();
    delete rng_;
  }

  template <typename Dtype>
  void ImageDepthLabelDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
  {
    // the function setups the following top data:
    // top[0] => image + depth + aux: [batch size] x [3] x [crop_size] x [crop_size] 
    // top[1] => image labels:  [batch size] x 1 x [crop_minus_margin_size] x [crop_minus_margin_size]  => this will go as-is to the image2mesh/loss/accuracy layers

    const ImageDepthLabelDataParameter &data_param = this->layer_param_.image_depth_label_data_param();
    rendered_image_dir_ = data_param.rendered_image_dir() + "/";
    depth_image_dir_ = data_param.depth_image_dir() + "/";
    aux_image_dir_ = data_param.aux_image_dir() + "/"; // SDF/UP change
    label_dir_ = data_param.label_dir() + "/";
    std::replace(rendered_image_dir_.begin(), rendered_image_dir_.end(), '\\', '/');
    std::replace(depth_image_dir_.begin(), depth_image_dir_.end(), '\\', '/');
    std::replace(aux_image_dir_.begin(), aux_image_dir_.end(), '\\', '/'); // SDF/UP change
    std::replace(label_dir_.begin(), label_dir_.end(), '\\', '/');

    validation_mode_ = data_param.validation_mode();
    batch_size_ = data_param.batch_size();
    image_margin_ = data_param.image_margin();
    image_mean_ = data_param.image_mean();
    depth_mean_ = data_param.depth_mean();
    aux_mean_ = data_param.aux_mean(); // SDF/UP change

    CHECK_GT(image_margin_, 0) << "Positive image size required";
    CHECK_GT(batch_size_, 0) << "Positive batch size required";
    CHECK_GE(validation_mode_, 0) << "Validation model can be 0, 1, or 2";
    CHECK_LE(validation_mode_, 2) << "Validation model can be 0, 1, or 2";

    // Find all images
    image_filename_lines_ = searchForImages(rendered_image_dir_);
    CHECK_GT(image_filename_lines_.size(), 0);

    // shuffle the list
    if (data_param.shuffle())
    {
      // randomly shuffle data
      LOG(INFO) << "Shuffling image list";
      const unsigned int prefetch_rng_seed = caffe_rng_rand();
      prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
      ShuffleImages(image_filename_lines_);
    }
    LOG(INFO) << "There is a total of " << image_filename_lines_.size() << " images.";

    // try to load a rendered image
    cv::Mat temp_cv_img = ReadImageToCVMat(rendered_image_dir_ + image_filename_lines_[0], false);
    CHECK(temp_cv_img.data) << "Could not load " << rendered_image_dir_ + image_filename_lines_[0];

    // get crop size from layer parameters
    crop_size_ = temp_cv_img.rows;
    const TransformationParameter& transform_param = this->layer_param_.transform_param();
    if (transform_param.has_crop_size())
      crop_size_ = transform_param.crop_size();
    LOG(INFO) << "Will use crop size " << crop_size_;
    crop_minus_margin_size_ = crop_size_ - 2 * image_margin_;
    CHECK_GT(crop_size_, 0) << "Crop size should have been specified as input parameter (and should have been positive)";
    CHECK_GT(crop_minus_margin_size_, 0) << "Crop size should have been bigger than 2 * margin size";

    // try to load a depth image
    string depth_image_filename = image_filename_lines_[0];
    boost::replace_all(depth_image_filename, "_int_", "_dep_");
    cv::Mat temp_cv_depth_img = ReadImageToCVMat(depth_image_dir_ + depth_image_filename, false);
    CHECK(temp_cv_depth_img.data) << "Could not load " << depth_image_dir_ + depth_image_filename;

    // try to load a aux image
    string aux_image_filename = image_filename_lines_[0];  // SDF/UP change
    boost::replace_all(aux_image_filename, "_int_", "_aux_");   // SDF/UP change
    cv::Mat temp_cv_aux_img = ReadImageToCVMat(aux_image_dir_ + aux_image_filename, false);   // SDF/UP change
    CHECK(temp_cv_aux_img.data) << "Could not load " << aux_image_dir_ + aux_image_filename;   // SDF/UP change

    // try to load a labels image
    string label_image_filename = image_filename_lines_[0];
    boost::replace_all(label_image_filename, "_int_", "_lbl_");
    cv::Mat temp_cv_label_img = ReadImageToCVMat(label_dir_ + label_image_filename, false);
    CHECK(temp_cv_label_img.data) << "Could not load " << label_dir_ + label_image_filename;


    // determine data blob shape
    data_shape[0] = batch_size_;
    data_shape[1] = 3; // 2=>3 SDF change
    data_shape[2] = crop_size_;
    data_shape[3] = crop_size_;
    top[0]->Reshape(data_shape);

    // determine label blob shape
    label_shape[0] = batch_size_;
    label_shape[1] = 1;
    label_shape[2] = crop_minus_margin_size_;  // last entry is used to save #faces per mesh
    label_shape[3] = crop_minus_margin_size_;
    top[1]->Reshape(label_shape);

    // prepare prefetch data
    for (int i = 0; i < this->PREFETCH_COUNT; ++i)
    {
      this->prefetch_[i].data_.Reshape(data_shape);
      this->prefetch_[i].label_.Reshape(label_shape);
    }

    // output info
    LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();

    LOG(INFO) << "output label size: " << top[1]->num() << ","
      << top[1]->channels() << "," << top[1]->height() << ","
      << top[1]->width();
  }



  template <typename Dtype>
  void ImageDepthLabelDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
    // setup timer 
    CPUTimer batch_timer;
    batch_timer.Start();
    double read_time = 0.0;
    double trans_time = 0.0;
    double preprocess_time = 0.0;
    CPUTimer timer;
    timer.Start();

    // layer params double check
    CHECK_GT(image_margin_, 0) << "Positive image size required";
    CHECK_GT(batch_size_, 0) << "Positive batch size required";
    CHECK_GT(crop_size_, 0) << "Crop size should have been specified as input parameter (and should have been positive)";
    CHECK_GT(crop_minus_margin_size_, 0) << "Crop size should have been bigger than 2 * margin size";
    CHECK_GT(image_filename_lines_.size(), 0) << "No training/test image filenames specified";

    // setup batch blob shapes
    batch->data_.Reshape(data_shape);
    batch->label_.Reshape(label_shape);

    // debug info
    DLOG(INFO) << "batch data size: " << batch->data_.num() << ","
      << batch->data_.channels() << "," << batch->data_.height() << ","
      << batch->data_.width();

    DLOG(INFO) << "batch label size: " << batch->label_.num() << ","
      << batch->label_.channels() << "," << batch->label_.height() << ","
      << batch->label_.width();

    // load all data into the batch
    size_t num_images = image_filename_lines_.size();
    preprocess_time = timer.MicroSeconds();
    for (int item_id = 0; item_id < batch_size_; ++item_id)
    {
      timer.Start();
      CHECK_GT(num_images, image_id_);

      // load rendered image
      if ( (image_filename_lines_[image_id_].at(image_filename_lines_[image_id_].size() - 5) == '@' && validation_mode_ == 0)
        || (image_filename_lines_[image_id_].at(image_filename_lines_[image_id_].size() - 5) != '@' && validation_mode_ == 2) )
      {
        item_id--;
        image_id_++;
        if (image_id_ >= num_images)
        {
          // We have reached the end. Restart from the first.
          DLOG(INFO) << "Restarting data prefetching from start.";
          image_id_ = 0;
          if (this->layer_param_.image_depth_label_data_param().shuffle())
            ShuffleImages(image_filename_lines_);
        }
        continue;
      }

      cv::Mat cv_img = ReadImageToCVMat(rendered_image_dir_ + image_filename_lines_[image_id_], false);
      CHECK(cv_img.data) << "Could not load " << rendered_image_dir_ + image_filename_lines_[image_id_];

      // try to load a depth image
      string depth_image_filename = image_filename_lines_[image_id_];
      boost::replace_all(depth_image_filename, "_int_", "_dep_");
      cv::Mat cv_depth_img = ReadImageToCVMat(depth_image_dir_ + depth_image_filename, false);
      CHECK(cv_depth_img.data) << "Could not load " << depth_image_dir_ + depth_image_filename;

      // try to load a aux image
      string aux_image_filename = image_filename_lines_[image_id_];  // SDF/UP change
      boost::replace_all(aux_image_filename, "_int_", "_aux_"); // SDF/UP change
      cv::Mat cv_aux_img = ReadImageToCVMat(aux_image_dir_ + aux_image_filename, false); // SDF/UP change
      CHECK(cv_aux_img.data) << "Could not load " << aux_image_dir_ + aux_image_filename; // SDF/UP change

      // load labels
      string label_image_filename = image_filename_lines_[image_id_];
      boost::replace_all(label_image_filename, "_int_", "_lbl_");
      cv::Mat cv_label_img = ReadImageToCVMat(label_dir_ + label_image_filename, false);
      CHECK(cv_label_img.data) << "Could not load " << label_dir_  + label_image_filename;

      switch (this->layer_param_.image_depth_label_data_param().padding())
      {
      case ImageDepthLabelDataParameter_Padding_ZERO:
        cv_img = ExtendLabelMargin(cv_img, image_margin_, image_margin_, 0);
        cv_depth_img = ExtendLabelMargin(cv_depth_img, image_margin_, image_margin_, 0);
        cv_aux_img = ExtendLabelMargin(cv_aux_img, image_margin_, image_margin_, 0); // SDF/UP change
        break;
      case ImageDepthLabelDataParameter_Padding_REFLECT:
        cv_img = ExtendLabelMargin(cv_img, image_margin_, image_margin_, -1);
        cv_depth_img = ExtendLabelMargin(cv_depth_img, image_margin_, image_margin_, -1);
        cv_aux_img = ExtendLabelMargin(cv_aux_img, image_margin_, image_margin_, -1); // SDF/UP change
        break;
      default:
        LOG(FATAL) << "Unknown Padding";
      }
      cv_label_img = ExtendLabelMargin(cv_label_img, image_margin_, image_margin_, 255);

      DLOG(INFO) << "Rendered image size: " << cv_img.size() << ", channels: " << cv_img.channels();
      DLOG(INFO) << "Depth image size: " << cv_depth_img.size() << ", channels: " << cv_depth_img.channels();
      DLOG(INFO) << "Aux image size: " << cv_aux_img.size() << ", channels: " << cv_aux_img.channels(); // SDF/UP change
      DLOG(INFO) << "Label image size: " << cv_label_img.size() << ", channels: " << cv_label_img.channels();

      read_time += timer.MicroSeconds();
      timer.Start();

      cv::Mat cv_transformed_img;
      cv::Mat cv_transformed_label_img;
      Transform(cv_img, cv_depth_img, cv_aux_img, cv_label_img, cv_transformed_img, cv_transformed_label_img); // SDF/UP change

      // rendered data => batch
      int mem_index = 0;
      int image_offset = batch->data_.offset(item_id);
      Dtype* transformed_data = batch->data_.mutable_cpu_data() + image_offset;
      for (int h = 0; h < crop_size_; ++h)
      {
        for (int w = 0; w < crop_size_; ++w)
        {
          for (int c = 0; c < 3; ++c) // 2=>3 SDF change
          {
            mem_index = (c * crop_size_ + h) * crop_size_ + w;
            cv::Vec3f pixel_values = cv_transformed_img.at<cv::Vec3f>(h, w); // SDF/UP change
            Dtype pixel_value = static_cast<Dtype>(pixel_values[c]);
            transformed_data[mem_index] = pixel_value;
          }
        }
      }

      int label_offset = batch->label_.offset(item_id);
      Dtype* transformed_label_data = batch->label_.mutable_cpu_data() + label_offset;
      // triangleid data => batch
      for (int h = 0; h < crop_minus_margin_size_; ++h)
      {
        for (int w = 0; w < crop_minus_margin_size_; ++w)
        {
          mem_index = h * crop_minus_margin_size_ + w;
          Dtype pixel_value = static_cast<Dtype>(cv_transformed_label_img.at<float>(h, w));
          transformed_label_data[mem_index] = pixel_value;
        }
      }

      trans_time += timer.MicroSeconds();

      // go to the next iter
      image_id_++;
      if (image_id_ >= num_images)
      {
        // We have reached the end. Restart from the first.
        DLOG(INFO) << "Restarting data prefetching from start.";
        image_id_ = 0;
        if (this->layer_param_.image_depth_label_data_param().shuffle())
          ShuffleImages(image_filename_lines_);
      }
    }

    batch_timer.Stop();
    DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
    DLOG(INFO) << "Preprocess time: " << preprocess_time / 1000 << " ms.";
    DLOG(INFO) << "      Read time: " << read_time / 1000 << " ms.";
    DLOG(INFO) << " Transform time: " << trans_time / 1000 << " ms.";
  }

  template <typename Dtype>
  cv::Mat ImageDepthLabelDataLayer<Dtype>::ExtendLabelMargin(cv::Mat &image, int margin_w, int margin_h, double value)
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
  vector<string> ImageDepthLabelDataLayer<Dtype>::searchForImages(const string& search_path)
  {
    CHECK(boost::filesystem::exists(search_path)) << "Directory " << search_path << " does not exist.";
    const std::regex pattern(".*\\.png");

    vector<string> matching_image_filenames;

    boost::filesystem::directory_iterator end_itr; // Default ctor yields past-the-end
    for (boost::filesystem::directory_iterator i(search_path); i != end_itr; ++i)
    {
      // Skip if not a file
      if (!boost::filesystem::is_regular_file(i->status())) continue;

      // Skip if no match:
      if (!std::regex_match(i->path().filename().string(), pattern)) continue;

      // File matches, store it
      matching_image_filenames.push_back(i->path().filename().string());
    }

    return matching_image_filenames;
  }


  template <typename Dtype>
  void ImageDepthLabelDataLayer<Dtype>::ShuffleImages(vector<string>& rendered_image_filenames)
  {
    caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
    vector<int> order(rendered_image_filenames.size());
    for (int i = 0; i < order.size(); ++i)
      order[i] = i;

    shuffle(order.begin(), order.end(), prefetch_rng);

    vector<std::string> new_rendered_image_filenames;
    for (int i = 0; i < order.size(); ++i)
    {
      string filename = rendered_image_filenames[order[i]];
      new_rendered_image_filenames.push_back(filename);
    }

    swap(rendered_image_filenames, new_rendered_image_filenames);
  }


  // perhaps this could into data_transformer, but for now let's leave it here
  template<typename Dtype>
  void ImageDepthLabelDataLayer<Dtype>::Transform(const cv::Mat& cv_img, const cv::Mat& cv_depth_img, const cv::Mat& cv_aux_img, const cv::Mat& cv_label_img, cv::Mat& cv_transformed_img, cv::Mat& cv_transformed_label_img) // SDF/UP change
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

    CHECK_EQ(cv_label_img.rows, img_height) << "Labels and rendered image should have the same height";
    CHECK_EQ(cv_label_img.cols, img_width) << "Labels and rendered image should have the same height";
    CHECK(cv_label_img.depth() == CV_8U) << "Label image type must be unsigned byte";
    CHECK(cv_label_img.channels() == 1) << "Label image should be made out of 1 channel";

    // cropping
    int h_off = 0;
    int w_off = 0;
    cv::Mat cv_cropped_img = cv_img.clone();
    cv::Mat cv_cropped_depth_img = cv_depth_img.clone();
    cv::Mat cv_cropped_aux_img = cv_aux_img.clone(); // SDF/UP change
    cv::Mat cv_cropped_label_img = cv_label_img.clone();
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
    cv_cropped_label_img = cv_label_img(roi);

    CHECK(cv_cropped_img.data);
    CHECK(cv_cropped_depth_img.data);
    CHECK(cv_cropped_aux_img.data); // SDF/UP change
    CHECK(cv_cropped_label_img.data);
    DLOG(INFO) << "Cropped image size: " << cv_cropped_img.size() << ", channels: " << cv_cropped_img.channels();
    DLOG(INFO) << "Cropped depth image size: " << cv_cropped_depth_img.size() << ", channels: " << cv_cropped_depth_img.channels();
    DLOG(INFO) << "Cropped aux image size: " << cv_cropped_aux_img.size() << ", channels: " << cv_cropped_aux_img.channels(); // SDF/UP change
    DLOG(INFO) << "Cropped label image size: " << cv_cropped_label_img.size() << ", channels: " << cv_cropped_label_img.channels();
    DLOG(INFO) << "Used height offset: " << h_off << ", width offset: " << w_off;

    cv::Mat cv_mirrored_cropped_img = cv_cropped_img.clone();
    cv::Mat cv_mirrored_cropped_depth_img = cv_cropped_depth_img.clone();
    cv::Mat cv_mirrored_cropped_aux_img = cv_cropped_aux_img.clone(); // SDF/UP change
    cv::Mat cv_mirrored_cropped_label_img = cv_cropped_label_img.clone();
    if (do_mirror)
    {
      cv::flip(cv_cropped_img, cv_mirrored_cropped_img, 1);
      cv::flip(cv_cropped_depth_img, cv_mirrored_cropped_depth_img, 1);
      cv::flip(cv_cropped_aux_img, cv_mirrored_cropped_aux_img, 1); // SDF/UP change
      cv::flip(cv_cropped_label_img, cv_mirrored_cropped_label_img, 1);
    }

    CHECK(cv_mirrored_cropped_img.data);
    CHECK(cv_mirrored_cropped_depth_img.data);
    CHECK(cv_mirrored_cropped_aux_img.data); // SDF/UP change
    CHECK(cv_mirrored_cropped_label_img.data);

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
    vector<cv::Mat> image_channels;
    image_channels.push_back(cv_mirrored_cropped_img);
    image_channels.push_back(cv_mirrored_cropped_depth_img); // note: if you want to ignore depth, replicate with cv_mirrored_cropped_img here
    image_channels.push_back(cv_mirrored_cropped_aux_img); // note: if you want to ignore aux, replicate with cv_mirrored_cropped_img here // SDF/UP change
    cv::merge(image_channels, cv_transformed_img);

    // unsigned char, 1 channel encoding label id, 1 float channel 
    DLOG(INFO) << "Label ID Image pixel (image_margin,image_margin) before cropping with margin: " << (int)cv_mirrored_cropped_label_img.at<uchar>(image_margin_, image_margin_);
    cv_transformed_label_img = cv::Mat(crop_minus_margin_size_, crop_minus_margin_size_, CV_32F);
    cv_transformed_label_img.setTo(255.0f); // default background pixels (ignore pixels), then fill
    for (int i = 0; i < crop_minus_margin_size_; ++i)
    {
      for (int j = 0; j < crop_minus_margin_size_; ++j)
      {
        cv_transformed_label_img.at<float>(i, j) = (float)cv_mirrored_cropped_label_img.at<uchar>(i + image_margin_, j + image_margin_);
      }
    }
    DLOG(INFO) << "Label ID Image pixel (image_margin,image_margin) after conversion/cropping with margin: " << cv_transformed_label_img.at<float>(0, 0);
  }


  INSTANTIATE_CLASS(ImageDepthLabelDataLayer);
  REGISTER_LAYER_CLASS(ImageDepthLabelData);

}  // namespace caffe