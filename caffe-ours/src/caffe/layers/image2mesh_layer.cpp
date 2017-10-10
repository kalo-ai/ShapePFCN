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

#include <cfloat>
#include <vector>

#include "caffe/layers/image2mesh_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void Image2MeshLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  // bottom[0] => mesh labels:  [batch size] x 1 x [max_num_faces] x 1
  // bottom[1] => triangle IDs (1 channel encoding triangle ids): [batch size] x [#views] x [crop_minus_margin_size] x [crop_minus_margin_size]
  // bottom[2...] => feature maps: [batch size] x [#labels] x [crop_minus_margin_size] x [crop_minus_margin_size] 
  // top[0] => unary feats:  [batch size] x [#labels] x [max_num_faces] x 1

  op_ = this->layer_param_.image2mesh_param().operation();
  for (int i = 3; i < bottom.size(); ++i) {
    CHECK(bottom[i]->shape() == bottom[2]->shape());
  }
  CHECK(bottom[2]->height() == bottom[1]->height());
  CHECK(bottom[2]->width() == bottom[1]->width());
  CHECK(bottom.size() - 2 == bottom[1]->channels());

  max_num_faces_ = bottom[0]->height(); // this is what the labels blob is used for 
  num_labels_ = bottom[2]->channels();
  num_pixels_ = bottom[1]->height() * bottom[1]->width();
}

template <typename Dtype>
void Image2MeshLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  const int batch_size = bottom[0]->num();
  vector<int> top_shape(4);
  top_shape[0] = batch_size;
  top_shape[1] = num_labels_;
  top_shape[2] = max_num_faces_;
  top_shape[3] = 1;
  top[0]->Reshape(top_shape);

  max_pixel_idx_.Reshape(top_shape);
  max_view_idx_.Reshape(top_shape);
}

template <typename Dtype>
void Image2MeshLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  const int batch_size = bottom[0]->num();

  // pointers to bottom/top data + indices
  const Dtype* bottom_feat_data = NULL;
  const Dtype* bottom_triangleid_data = NULL;
  Dtype* top_data = top[0]->mutable_cpu_data();
  int* max_pixel_idx = max_pixel_idx_.mutable_cpu_data();
  int* max_view_idx = max_view_idx_.mutable_cpu_data();

  // initialize top feats as uniform, index to -1
  const int top_data_size = top[0]->count();
  if (op_ == Image2MeshParameter_Image2MeshOp_MAX)
  {
    caffe_set(top_data_size, Dtype(-FLT_MAX), top_data);
  }
  else
  {
    caffe_set(top_data_size, Dtype(0.0), top_data);
  }
  caffe_set(top_data_size, -1, max_pixel_idx );
  caffe_set(top_data_size, -1, max_view_idx);

  // parallelism?
  for (int item_id = 0; item_id < batch_size; ++item_id)
  {
    for (int view_id = 0; view_id < bottom.size() - 2; ++view_id)
    {
      bottom_triangleid_data = bottom[1]->cpu_data() + bottom[1]->offset(item_id, view_id);

      for (int label_id = 0; label_id < num_labels_; ++label_id)
      {
        bottom_feat_data = bottom[view_id + 2]->cpu_data() + bottom[view_id + 2]->offset(item_id, label_id);
        top_data = top[0]->mutable_cpu_data() + top[0]->offset(item_id, label_id);
        max_pixel_idx = max_pixel_idx_.mutable_cpu_data() + max_pixel_idx_.offset(item_id, label_id);
        max_view_idx = max_view_idx_.mutable_cpu_data() + max_view_idx_.offset(item_id, label_id);

        for (int pixel_id = 0; pixel_id < num_pixels_; ++pixel_id)
        {
          int triangle_id = (int)bottom_triangleid_data[pixel_id];
          if (triangle_id < 0) // background pixel
            continue;

          if (op_ == Image2MeshParameter_Image2MeshOp_MAX)
          {
            if (bottom_feat_data[pixel_id] > top_data[triangle_id])
            {
              top_data[triangle_id] = bottom_feat_data[pixel_id];
              max_pixel_idx[triangle_id] = pixel_id;
              max_view_idx[triangle_id] = view_id;
            }
          }
          else
          {
            top_data[triangle_id] += bottom_feat_data[pixel_id]; 
          }
        }  // end of loop over pixel ids
      } // end of loop over label ids
    } // end of loop over view ids
  } // end of loop over item ids (meshes) in batch 
}

template <typename Dtype>
void Image2MeshLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
  const int batch_size = bottom[0]->num();

  const Dtype* top_diff = NULL;
  Dtype* bottom_diff = NULL;
  const Dtype* bottom_triangleid_data = NULL;
  const int* max_pixel_idx = max_pixel_idx_.mutable_cpu_data();
  const int* max_view_idx = max_view_idx_.mutable_cpu_data();

  // parallelism?
  for (int item_id = 0; item_id < batch_size; ++item_id)
  {
    for (int view_id = 0; view_id < bottom.size() - 2; ++view_id)
    {
      if (!propagate_down[view_id + 2])
        continue;
      bottom_triangleid_data = bottom[1]->cpu_data() + bottom[1]->offset(item_id, view_id);

      for (int label_id = 0; label_id < num_labels_; ++label_id)
      {
        bottom_diff = bottom[view_id + 2]->mutable_cpu_diff() + bottom[view_id + 2]->offset(item_id, label_id);
        caffe_set(num_pixels_, Dtype(0), bottom_diff);
        top_diff = top[0]->cpu_diff() + top[0]->offset(item_id, label_id);
        max_pixel_idx = max_pixel_idx_.cpu_data() + max_pixel_idx_.offset(item_id, label_id);
        max_view_idx = max_view_idx_.cpu_data() + max_view_idx_.offset(item_id, label_id);

        for (int pixel_id = 0; pixel_id < num_pixels_; ++pixel_id)
        {
          int triangle_id = (int)bottom_triangleid_data[pixel_id];
          if (triangle_id == -1)
            continue;

          if (op_ == Image2MeshParameter_Image2MeshOp_MAX)
          {
            if (max_pixel_idx[triangle_id] == pixel_id && max_view_idx[triangle_id] == view_id)
              bottom_diff[pixel_id] += top_diff[triangle_id]; 
          }
          else
          {
            bottom_diff[pixel_id] += top_diff[triangle_id];
          }
        }  // end of loop over pixel ids
      } // end of loop over label ids
    } // end of loop over view ids
  } // end of loop over item ids (meshes) in batch 
}

INSTANTIATE_CLASS(Image2MeshLayer);
REGISTER_LAYER_CLASS(Image2Mesh);

}  // namespace caffe
