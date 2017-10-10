#include <boost/thread.hpp>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

template <typename Dtype>
BaseDataLayer<Dtype>::BaseDataLayer(const LayerParameter& param, const bool  _output_labels_, const bool _output_image2mesh_data_, const bool _output_crfpairwise_data_)
  : Layer<Dtype>(param),
  transform_param_(param.transform_param()),
  output_labels_(_output_labels_),
  output_image2mesh_data_(_output_image2mesh_data_),
  output_crfpairwise_data_(_output_crfpairwise_data_)
{
}

template <typename Dtype>
void BaseDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  data_transformer_.reset( new DataTransformer<Dtype>(transform_param_, this->phase_));
  data_transformer_->InitRand();
  // The subclasses should setup the size of bottom and top
  DataLayerSetUp(bottom, top);
}

template <typename Dtype>
BasePrefetchingDataLayer<Dtype>::BasePrefetchingDataLayer(const LayerParameter& param, const bool  _output_labels_, const bool _output_image2mesh_data_, const bool _output_crfpairwise_data_)
  : BaseDataLayer<Dtype>(param, _output_labels_, _output_image2mesh_data_, _output_crfpairwise_data_),
      prefetch_free_(), prefetch_full_() {
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
    prefetch_free_.push(&prefetch_[i]);
  }
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);
  // Before starting the prefetch thread, we make cpu_data and gpu_data
  // calls so that the prefetch thread does not accidentally make simultaneous
  // cudaMalloc calls when the main thread is running. In some GPUs this
  // seems to cause failures if we do not so.
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
    prefetch_[i].data_.mutable_cpu_data();
    if (this->output_image2mesh_data_)
      prefetch_[i].image2mesh_data_.mutable_cpu_data();
    if (this->output_crfpairwise_data_)
      prefetch_[i].crfpairwise_data_.mutable_cpu_data();
    if (this->output_labels_)
      prefetch_[i].label_.mutable_cpu_data();    
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    for (int i = 0; i < PREFETCH_COUNT; ++i) 
    {
      prefetch_[i].data_.mutable_gpu_data();
      if (this->output_image2mesh_data_)
        prefetch_[i].image2mesh_data_.mutable_gpu_data();
      if (this->output_crfpairwise_data_)
        prefetch_[i].crfpairwise_data_.mutable_gpu_data();
      if (this->output_labels_)
        prefetch_[i].label_.mutable_gpu_data();
    }
  }
#endif
  DLOG(INFO) << "Initializing prefetch";
  this->data_transformer_->InitRand();
  StartInternalThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::InternalThreadEntry() {
#ifndef CPU_ONLY
  cudaStream_t stream;
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
#endif

  try {
    while (!must_stop()) {
      Batch<Dtype>* batch = prefetch_free_.pop();
      load_batch(batch);
#ifndef CPU_ONLY
      if (Caffe::mode() == Caffe::GPU) {
        batch->data_.data().get()->async_gpu_push(stream);
        if (this->output_image2mesh_data_)
          batch->image2mesh_data_.data().get()->async_gpu_push(stream);
        if (this->output_crfpairwise_data_)
          batch->crfpairwise_data_.data().get()->async_gpu_push(stream);
        if (this->output_labels_)
          batch->label_.data().get()->async_gpu_push(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
#endif
      prefetch_full_.push(batch);
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
#endif
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.cpu_data(), top[0]->mutable_cpu_data());
  DLOG(INFO) << "Prefetch copied";

  if (this->output_image2mesh_data_)
  {
    top[1]->ReshapeLike(batch->image2mesh_data_);
    caffe_copy(batch->image2mesh_data_.count(), batch->image2mesh_data_.cpu_data(), top[1]->mutable_cpu_data());
  }

  if (this->output_crfpairwise_data_)
  {
    int top_id = 1;
    if (this->output_image2mesh_data_)
      top_id++;
    top[top_id]->ReshapeLike(batch->crfpairwise_data_);
    caffe_copy(batch->crfpairwise_data_.count(), batch->crfpairwise_data_.cpu_data(), top[top_id]->mutable_cpu_data());
  } 

  if (this->output_labels_) 
  {
    int top_id = 1;
    if (this->output_image2mesh_data_)
      top_id++;
    if (this->output_crfpairwise_data_)
      top_id++;
    // Reshape to loaded labels.
    top[top_id]->ReshapeLike(batch->label_);
    caffe_copy(batch->label_.count(), batch->label_.cpu_data(), top[top_id]->mutable_cpu_data());
  }

  prefetch_free_.push(batch);
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(BasePrefetchingDataLayer, Forward);
#endif

INSTANTIATE_CLASS(BaseDataLayer);
INSTANTIATE_CLASS(BasePrefetchingDataLayer);

}  // namespace caffe
