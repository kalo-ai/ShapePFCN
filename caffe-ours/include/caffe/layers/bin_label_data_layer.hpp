#ifndef CAFFE_BIN_LABEL_DATA_LAYER_H
#define CAFFE_BIN_LABEL_DATA_LAYER_H

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template<typename Dtype>
class BinLabelDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit BinLabelDataLayer(const LayerParameter &param)
      : BasePrefetchingDataLayer<Dtype>(param, true) { }

  virtual ~BinLabelDataLayer();

  virtual void DataLayerSetUp(const vector<Blob<Dtype> *> &bottom,
                              const vector<Blob<Dtype> *> &top);

  virtual inline const char *type() const { return "BinLabelData"; }

  virtual inline int ExactNumBottomBlobs() const { return 0; }

  virtual inline int ExactNumTopBlobs() const { return -1; }

  virtual inline int MaxTopBlobs() const { return 3; }

  virtual inline int MinTopBlobs() const { return 2; }

  int Rand(int n);

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;

  virtual void ShuffleImages();

  virtual void load_batch(Batch<Dtype> *batch);

  std::vector<std::string> bin_names_;
  std::vector<std::string> label_names_;
  int lines_id_;

  shared_ptr<Caffe::RNG> rng_;
};

} // namespace caffe

#endif //CAFFE_BIN_LABEL_DATA_LAYER_H
