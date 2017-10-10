#include <opencv2/core/core.hpp>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "caffe/layers/bin_label_data_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"


using std::string;
using std::vector;

namespace {

template <typename Dtype>
void PadImage(const cv::Mat &image, int height, int width, double value,
              Dtype *data) {
  CHECK_EQ(image.type(), CV_32F);
  cv::Mat output_image(height, width, CV_32F, data);
  int input_height = image.rows;
  int input_width = image.cols;
  int top, bottom, left, right;
  top = (height - input_height) / 2;
  bottom = height - input_height - top;
  left = (width - input_width) / 2;
  right = width - input_width - left;
  cv::copyMakeBorder(image, output_image, top, bottom, left, right,
                     cv::BORDER_CONSTANT, cv::Scalar(value));
}


template <typename Dtype>
void ExtendImage(const cv::Mat &image, int rows, int cols,
                 int row_margin, int col_margin, double value,
                 Dtype*data) {
  CHECK_EQ(image.type(), CV_32F);
  int input_height = image.rows;
  int input_width = image.cols;
  int output_rows = rows + row_margin * 2;
  int output_cols = cols + col_margin * 2;
  int top, bottom, left, right;
  top = row_margin;
  bottom = output_rows - top - input_height;
  left = col_margin;
  right = output_cols - left - input_width;
  cv::Mat output_image(output_rows, output_cols, CV_32F, data);
  if (value >= 0) {
    cv::copyMakeBorder(image, output_image, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(value));
  } else {
    cv::copyMakeBorder(image, output_image, top, bottom, left, right,
                       cv::BORDER_REFLECT_101);
  }
}


int ReadBinImageChannels(const std::string &filename) {
  int type_code;
  int shape_size;
  int count;
  std::vector<int> shape;
  FILE *fp = fopen(filename.c_str(), "rb");
  CHECK(fp != NULL) << "Failed to open " << filename;
  count = fread((void*)(&type_code), sizeof(int), 1, fp);
  CHECK_EQ(count, 1);
  count = fread((void*)(&shape_size), sizeof(int), 1, fp);
  CHECK_EQ(count, 1);
  shape.resize(shape_size);
  count = fread((void*)shape.data(), sizeof(int), shape_size, fp);
  CHECK_EQ(count, shape_size);
  fclose(fp);
  return shape[0];
}


cv::Mat ReadImage(const std::string &filename) {
  int type_code;
  int shape_size;
  int count;
  std::vector<int> shape;
  FILE *fp = fopen(filename.c_str(), "rb");
  CHECK(fp != NULL) << "Failed to open " << filename;
  count = fread((void*)(&type_code), sizeof(int), 1, fp);
  CHECK_EQ(count, 1);
  count = fread((void*)(&shape_size), sizeof(int), 1, fp);
  CHECK_EQ(count, 1);
  shape.resize(shape_size);
  count = fread((void*)shape.data(), sizeof(int), shape_size, fp);
  CHECK_EQ(count, shape_size);

  cv::Mat image(shape_size, shape.data(), type_code);
  count = fread((void*)image.data, image.elemSize1(),
                image.total(), fp);
  CHECK_EQ(count, image.total()) << "file: " << filename
  << " type: " << type_code << " shape size: " << shape_size
  << " shape: " << shape[0] << ' ' << shape[1] << ' '
  << shape[2];
  fclose(fp);

  return image;
}


vector<int> ReadImageShape(const std::string &filename) {
  int type_code;
  int shape_size;
  int count;
  std::vector<int> shape;
  FILE *fp = fopen(filename.c_str(), "rb");
  CHECK(fp != NULL) << "Failed to open " << filename;
  count = fread((void*)(&type_code), sizeof(int), 1, fp);
  CHECK_EQ(count, 1);
  count = fread((void*)(&shape_size), sizeof(int), 1, fp);
  CHECK_EQ(count, 1);
  shape.resize(shape_size);
  count = fread((void*)shape.data(), sizeof(int), shape_size, fp);
  CHECK_EQ(count, shape_size);
  fclose(fp);
  return shape;
}

template <typename Dtype>
void MirrorImage(cv::Mat &image) {
  CHECK_EQ(image.type(), CV_32F);
  int channels, height, width;
  if (image.dims == 2) {
    channels = 1;
    height = image.rows;
    width = image.cols;
  } else {
    channels = image.size[0];
    height = image.size[1];
    width = image.size[2];
  }
  Dtype *data = reinterpret_cast<Dtype*>(image.data);
  Dtype tmp;
  int idx0, idx1;
  for (int c = 0; c < channels; ++c) {
    Dtype *channel_data = data + c * height * width;
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width / 2; ++w) {
        idx0 = h * width + w;
        idx1 = h * width + width - w - 1;
        tmp = channel_data[idx0];
        channel_data[idx0] = channel_data[idx1];
        channel_data[idx1] = tmp;
      }
    }
  }
}


//cv::Mat SliceImage(const cv::Mat &image, int height, int width) {
//  CHECK_EQ(image.dims, 3);
//  std::vector<cv::Range> ranges(3);
//  ranges[0] = cv::Range::all();
//  ranges[1] = cv::Range(0, height);
//  ranges[2] = cv::Range(0, width);
//  return image(ranges.data()).clone();
//}

cv::Mat SliceMatrix(const cv::Mat &image, const caffe::Slice &slice,
                    int pad = 255) {
  int offset[2] = {0, 0};
  int stride[2] = {1, 1};
  if (slice.offset_size() == 1) {
    offset[0] = offset[1] = slice.offset(0);
  } else if (slice.offset_size() > 1) {
    offset[0] = slice.offset(0);
    offset[1] = slice.offset(1);
  }
  if (slice.stride_size() == 1) {
    stride[0] = stride[1] = slice.stride(0);
  } else if (slice.stride_size() > 1) {
    stride[0] = slice.stride(0);
    stride[1] = slice.stride(1);
  }
  CHECK_EQ(image.type(), CV_32F);
//  CHECK_GE(image.rows, offset[0] + stride[0] * slice.dim(0));
//  CHECK_GE(image.cols, offset[1] + stride[1] * slice.dim(1));
  cv::Mat out_image(slice.dim(0), slice.dim(1), CV_32F);
  for (int r = 0; r < slice.dim(0); ++r) {
    int input_row = r * stride[0] + offset[0];
    if (input_row >= image.rows) {
      for (int c = 0; c < slice.dim(1); ++c) {
        out_image.at<float>(r, c) = pad;
      }
    } else {
      for (int c = 0; c < slice.dim(1); ++c) {
        int input_col = c * stride[1] + offset[1];
        if (input_col >= image.cols) {
          out_image.at<float>(r, c) = pad;
        } else {
          out_image.at<float>(r, c) = image.at<float>(input_row, input_col);
        }
      }
    }
  }
  return out_image;
}

cv::Mat SliceChannels(const cv::Mat &data, const caffe::Slice &slice) {
  if (slice.dim_size() + slice.stride_size() + slice.offset_size() == 0) {
    return data;
  }
  int offset[2] = {0, 0};
  int stride[2] = {1, 1};
  int dims[2] = {data.size[1], data.size[2]};
  if (slice.offset_size() == 1) {
    offset[0] = offset[1] = slice.offset(0);
  } else if (slice.offset_size() > 1) {
    offset[0] = slice.offset(0);
    offset[1] = slice.offset(1);
  }
  if (slice.stride_size() == 1) {
    stride[0] = stride[1] = slice.stride(0);
  } else if (slice.stride_size() > 1) {
    stride[0] = slice.stride(0);
    stride[1] = slice.stride(1);
  }
  if (slice.dim_size() == 1) {
    dims[0] = dims[1] = slice.dim(0);
  } else {
    dims[0] = slice.dim(0);
    dims[1] = slice.dim(1);
  }
  vector<int> out_sizes(3);
  out_sizes[0] = data.size[0];
  out_sizes[1] = dims[0];
  out_sizes[2] = dims[1];
  cv::Mat out_image(out_sizes.size(), out_sizes.data(), CV_32F);
  for (int i = 0; i < out_sizes[0]; ++i) {
    for (int r = 0; r < dims[0]; ++r) {
      for (int c = 0; c < dims[1]; ++c) {
        out_image.at<float>(i, r, c) = data.at<float>(
            i, r * stride[0] + offset[0], c * stride[1] + offset[1]);
      }
    }
  }
  return out_image;
}

//void ReadImage(const std::string& filename, cv::Mat *image) {
//  int type_code;
//  int shape_size;
//  int count;
//  vector<int> shape;
//  FILE *fp = fopen(filename.c_str(), "rb");
//  CHECK(fp != NULL) << "Failed to open " << filename;
//  count = fread((void*)(&type_code), sizeof(int), 1, fp);
//  CHECK_EQ(count, 1);
//  count = fread((void*)(&shape_size), sizeof(int), 1, fp);
//  CHECK_EQ(count, 1);
//  shape->resize(shape_size);
//  count = fread((void*)shape.data(), sizeof(int), shape_size, fp);
//  CHECK_EQ(count, shape_size);
//
////  int channels = (*shape)[0]
////  int height = (*shape)[1];
////  int width = (*shape)[2];
//  // CHECK_EQ(type_code, CV_32F);
//  // CHECK_EQ(shape_size, 3);
//
//  image->create(shape_size, shape.data(), type_code);
//  int data_size = image->total() * image->elemSize();
//  count = fread((void*)image->data, 1, data_size, fp);
//  fclose(fp);
//  CHECK_EQ(data_size, count);
//}


template <typename Dtype>
void ReadBinImage(const std::string& filename, int channels, const int height, const int width,
                  Dtype *data) {
  int type_code;
  int shape_size;
  int count;
  std::vector<int> shape;
  FILE *fp = fopen(filename.c_str(), "rb");
  CHECK(fp != NULL) << "Failed to open " << filename;
  count = fread((void*)(&type_code), sizeof(int), 1, fp);
  CHECK_EQ(count, 1);
  count = fread((void*)(&shape_size), sizeof(int), 1, fp);
  CHECK_EQ(count, 1);
  shape.resize(shape_size);
  count = fread((void*)shape.data(), sizeof(int), shape_size, fp);
  CHECK_EQ(count, shape_size);

  int input_height = shape[1];
  int input_width = shape[2];
  CHECK_EQ(type_code, CV_32F);
  CHECK_EQ(shape_size, 3);
  CHECK_EQ(channels, shape[0]);
  CHECK_GE(height, input_height);
  CHECK_GE(width, input_width);

  int input_size = input_height * input_width;
  cv::Mat input_image(input_height, input_width, type_code);
  for (int c = 0; c < channels; ++c) {
    count = fread((void*)input_image.data, sizeof(float), input_size, fp);
    CHECK_EQ(count, input_size);
    PadImage(input_image, height, width, 0, data);
    data += height * width;
  }
  fclose(fp);
}

template <typename Dtype>
void AssignEvenLabelWeight(const Dtype *labels, int num, Dtype *weights) {
  Dtype max_label = 0;
  for (int i = 0; i < num; ++i) {
    if (labels[i] != 255) {
      max_label = std::max(labels[i], max_label);
    }
  }
  int num_labels = static_cast<int>(max_label) + 1;
  std::vector<int> counts(num_labels, 0);
  std::vector<double> label_weight(num_labels);
  Dtype total = 0;
  for (int i = 0; i < num; ++i) {
    if (labels[i] != 255) {
      counts[static_cast<int>(labels[i])] += 1;
      total += 1;
    }
  }
  for (int i = 0; i < num_labels; ++i) {
    if (counts[i] == 0) {
      label_weight[i] = 0;
    } else {
      label_weight[i] = total / counts[i];
    }
    // printf("%d weight %f\n", i, label_weight[i]);
  }
  for (int i = 0; i < num; ++i) {
    weights[i] = label_weight[static_cast<int>(labels[i])];
  }
}

}

namespace caffe {

template <typename Dtype>
BinLabelDataLayer<Dtype>::~BinLabelDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void BinLabelDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  auto &data_param = this->layer_param_.bin_label_data_param();

// Read the file with filenames and labels
  const string& bin_list_path = data_param.bin_list_path();
  auto &label_slice = data_param.label_slice();
  LOG(INFO) << "Opening bin list " << bin_list_path;
  std::ifstream infile(bin_list_path.c_str());
  string filename;
  while (infile >> filename) {
    bin_names_.push_back(filename);
  }
  infile.close();

  const string& label_list_path = data_param.label_list_path();
  LOG(INFO) << "Opening label list " << label_list_path;
  infile.open(label_list_path.c_str());
  while (infile >> filename) {
    label_names_.push_back(filename);
  }
  infile.close();

  CHECK_EQ(bin_names_.size(), label_names_.size());

  if (data_param.shuffle()) {
// randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << bin_names_.size() << " images.";

  lines_id_ = 0;

  vector<int> data_shape(4);
  // vector<int> bin_shape = ReadImageShape(data_param.bin_dir() + bin_names_[0]);
  cv::Mat bin_image = SliceChannels(
      ReadImage(data_param.bin_dir() + bin_names_[0]), data_param.bin_slice());
  const int *bin_shape = bin_image.size;
  const int batch_size = data_param.batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  data_shape[0] = batch_size;
  data_shape[1] = bin_shape[0];
  data_shape[2] = bin_shape[1];
  data_shape[3] = bin_shape[2];
  top[0]->Reshape(data_shape);

  vector<int> label_shape(4);
  label_shape[0] = batch_size;
  label_shape[1] = 1;
  label_shape[2] = label_slice.dim(0);
  label_shape[3] = label_slice.dim(1);
  top[1]->Reshape(label_shape);

  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(data_shape);
    this->prefetch_[i].label_.Reshape(label_shape);
  }

  LOG(INFO) << "output data size: " << top[0]->num() << ","
  << top[0]->channels() << "," << top[0]->height() << ","
  << top[0]->width();

  LOG(INFO) << "output label size: " << top[1]->num() << ","
  << top[1]->channels() << "," << top[1]->height() << ","
  << top[1]->width();

  const unsigned int rng_seed = caffe_rng_rand();
  rng_.reset(new Caffe::RNG(rng_seed));
}

template <typename Dtype>
void BinLabelDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  vector<int> order(bin_names_.size());
  for (int i = 0; i < order.size(); ++i) {
    order[i] = i;
  }
  shuffle(order.begin(), order.end(), prefetch_rng);
  vector<std::string> new_image_lines(bin_names_.size());
  vector<std::string> new_label_lines(label_names_.size());
  for (int i = 0; i < order.size(); ++i) {
    new_image_lines[i] = bin_names_[order[i]];
    new_label_lines[i] = label_names_[order[i]];
  }
  swap(bin_names_, new_image_lines);
  swap(label_names_, new_label_lines);
}


template <typename Dtype>
int BinLabelDataLayer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}


// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void BinLabelDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  auto &data_param = this->layer_param_.bin_label_data_param();
  const int batch_size = data_param.batch_size();
  const string &bin_dir = data_param.bin_dir();
  const string &label_dir = data_param.label_dir();
  caffe::Slice label_slice = data_param.label_slice();

  // vector<int> bin_shape = ReadImageShape(bin_dir + bin_names_[lines_id_]);
  cv::Mat bin_image = SliceChannels(
      ReadImage(data_param.bin_dir() + bin_names_[lines_id_]),
      data_param.bin_slice());
  const int *bin_shape = bin_image.size;

  int output_height = bin_shape[1];
  int output_width = bin_shape[2];

  vector<int> data_shape(4);
  data_shape[0] = batch_size;
  data_shape[1] = bin_shape[0];
  data_shape[2] = output_height;
  data_shape[3] = output_width;
  // this->transformed_data_.Reshape(data_shape);
  batch->data_.Reshape(data_shape);

  vector<int> label_shape(4);
  label_shape[0] = batch_size;
  label_shape[1] = 1;
  label_shape[2] = label_slice.dim(0);
  label_shape[3] = label_slice.dim(1);
  batch->label_.Reshape(label_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  auto lines_size = bin_names_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    bool do_mirror = data_param.mirror() && Rand(2);
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image

    int image_offset = batch->data_.offset(item_id);
    int label_offset = batch->label_.offset(item_id);

//    ReadBinImage(image_dir + image_ids_[lines_id_] + image_ext, channels,
//                 new_height, new_width, prefetch_data + image_offset);
    std::string image_path = bin_dir + bin_names_[lines_id_];
    std::string label_path = label_dir + label_names_[lines_id_];
    DLOG(INFO) << "Reading " << image_path << std::endl;
    cv::Mat image = ReadImage(image_path);
    image = SliceChannels(image, data_param.bin_slice());
    // std::cout << "Reading " << label_path << std::endl;
    cv::Mat cv_label = cv::imread(label_path, 0);
    CHECK_GT(cv_label.total(), 0);
    cv_label.convertTo(cv_label, CV_32F);
    cv_label = SliceMatrix(cv_label, label_slice);

    if (do_mirror) {
      MirrorImage<Dtype>(image);
      MirrorImage<Dtype>(cv_label);
    }

    cv::Mat out_label(2, label_shape.data() + 2, cv::DataType<Dtype>::type,
                      prefetch_label + label_offset);
    CHECK_EQ(cv_label.total(), out_label.total());
    CHECK_EQ(cv_label.type(), out_label.type());
    cv_label.copyTo(out_label);

    cv::Mat out_data(3, data_shape.data() + 1, cv::DataType<Dtype>::type,
                     prefetch_data + image_offset);
    CHECK_EQ(image.total(), out_data.total());
    CHECK_EQ(image.type(), out_data.type());
    image.copyTo(out_data);

//     PadImage(cv_label, new_height, new_width, 255, label_data);
//    ExtendImage(cv_label, new_height, new_width, 0, 0, 255,
//                label_data);

//    Dtype *image_data = prefetch_data + image_offset;
//    for (int c = 0; c < channels; ++c) {
//      cv::Mat channel(label_height, label_width, CV_32F, image.ptr<float>(c));
////      PadImage(channel, new_height, new_width, 0,
////               image_data + c * new_height * new_width);
//      switch (data_param.padding()) {
//        case BinLabelDataParameter_Padding_ZERO:
//          ExtendImage(channel, new_height, new_width, margin_h, margin_w, 0,
//                      image_data + c * output_height * output_width);
//          break;
//        case BinLabelDataParameter_Padding_REFLECT:
//          ExtendImage(channel, new_height, new_width, margin_h, margin_w, -1,
//                      image_data + c * output_height * output_width);
//          break;
//        default:
//          LOG(FATAL) << "Unknown Padding";
//      }
//    }

    trans_time += timer.MicroSeconds();

    // prefetch_label[item_id] = lines_[lines_id_].second;
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (data_param.shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(BinLabelDataLayer);
REGISTER_LAYER_CLASS(BinLabelData);

}  // namespace caffe
