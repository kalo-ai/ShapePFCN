#include <vector>

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.gpu_data(), top[0]->mutable_gpu_data());

  if (this->output_image2mesh_data_)
  {
    top[1]->ReshapeLike(batch->image2mesh_data_);
    caffe_copy(batch->image2mesh_data_.count(), batch->image2mesh_data_.gpu_data(), top[1]->mutable_gpu_data());
  }

  if (this->output_crfpairwise_data_)
  {
    int top_id = 1;
    if (this->output_image2mesh_data_)
      top_id++;
    top[top_id]->ReshapeLike(batch->crfpairwise_data_);
    caffe_copy(batch->crfpairwise_data_.count(), batch->crfpairwise_data_.gpu_data(), top[top_id]->mutable_gpu_data());
  }

  if (this->output_labels_)
  {
    int top_id = 1;
    if (this->output_image2mesh_data_)
      top_id++;
    // Reshape to loaded labels.
    if (this->output_crfpairwise_data_)
      top_id++;
    top[top_id]->ReshapeLike(batch->label_);
    caffe_copy(batch->label_.count(), batch->label_.gpu_data(), top[top_id]->mutable_gpu_data());
  }

  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  prefetch_free_.push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);

}  // namespace caffe
