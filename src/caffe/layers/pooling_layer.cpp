#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template<typename Dtype, typename MItype, typename MOtype>
void PoolingLayer<Dtype, MItype, MOtype>::LayerSetUp(
                                        const vector<Blob<MItype>*>& bottom,
                                        const vector<Blob<MOtype>*>& top) {
  PoolingParameter pool_param = this->layer_param_.pooling_param();

  // Set the max number of top blobs before calling base Layer::SetUp.
  // If doing MAX pooling, we can optionally output an extra top Blob
  // for the mask.  Otherwise, we only have one top Blob.
  if (pool_param.pool() == PoolingParameter_PoolMethod_MAX) {
    max_top_blobs_ = 2;
  } else {
    max_top_blobs_ = 1;
  }

  channel_axis_ = bottom[0]->CanonicalAxisIndex(pool_param.axis());
  channels_ = bottom[0]->shape(channel_axis_);

  const int_tp first_spatial_axis = channel_axis_ + 1;
  const int_tp num_axes = bottom[0]->num_axes();
  num_spatial_axes_ = num_axes - first_spatial_axis;
  CHECK_GE(num_spatial_axes_, 0);

  vector<int_tp> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  vector<int_tp> spatial_dim_blob_shape(
      1, std::max(num_spatial_axes_, (int_tp) 1));

  kernel_shape_.Reshape(spatial_dim_blob_shape);
  int_tp* kernel_shape_data = kernel_shape_.mutable_cpu_data();

  if (pool_param.global_pooling()) {
    global_pooling_ = true;
    CHECK(!((pool_param.kernel_size_size() > 0) ||
      pool_param.has_kernel_h() || pool_param.has_kernel_w()))
      << "With Global_pooling: true Filter size cannot specified";
  } else {
    global_pooling_ = false;
    CHECK(!(pool_param.kernel_size_size() > 0) !=
        !(pool_param.has_kernel_h() && pool_param.has_kernel_w()))
        << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
    CHECK((pool_param.kernel_size_size() > 0) ||
        (pool_param.has_kernel_h() && pool_param.has_kernel_w()))
        << "For non-square filters both kernel_h and kernel_w are required.";
    if (pool_param.has_kernel_h() && pool_param.has_kernel_w()) {
      kernel_shape_data[0] = pool_param.kernel_h();
      kernel_shape_data[1] = pool_param.kernel_w();
    } else {
      const int_tp num_kernel_dims = pool_param.kernel_size_size();
      CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_);
      for (int_tp i = 0; i < num_spatial_axes_; ++i) {
        kernel_shape_data[i] = pool_param.kernel_size(
            (num_kernel_dims == 1) ? 0 : i);
        CHECK_GT(kernel_shape_data[i], 0)
          << "Filter dimensions must be nonzero.";
      }
    }
  }
  CHECK((!pool_param.has_pad() && pool_param.has_pad_h()
      && pool_param.has_pad_w())
      || (!pool_param.has_pad_h() && !pool_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!pool_param.has_stride() && pool_param.has_stride_h()
      && pool_param.has_stride_w())
      || (!pool_param.has_stride_h() && !pool_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  global_pooling_ = pool_param.global_pooling();
  round_mode_ = pool_param.round_mode();
  if (global_pooling_) {
    for (int_tp i = 0; i < num_spatial_axes_; ++i) {
      kernel_shape_data[i] = size_data[i];
    }
  }

  // Setup stride dimensions (stride_).
  stride_.Reshape(spatial_dim_blob_shape);
  int_tp* stride_data = stride_.mutable_cpu_data();
  if (pool_param.has_stride_h() || pool_param.has_stride_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "stride_h & stride_w can only be used for 2D convolution.";
    CHECK_EQ(0, pool_param.stride_size())
        << "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = pool_param.stride_h();
    stride_data[1] = pool_param.stride_w();
  } else {
    const int_tp num_stride_dims = pool_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == num_spatial_axes_)
        << "stride must be specified once, or once per spatial dimension "
        << "(stride specified " << num_stride_dims << " times; "
        << num_spatial_axes_ << " spatial dims);";
    const int_tp kDefaultStride = 1;
    for (int_tp i = 0; i < num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
          pool_param.stride((num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }

  // Setup pad dimensions (pad_).
  pad_.Reshape(spatial_dim_blob_shape);
  int_tp* pad_data = pad_.mutable_cpu_data();
  if (pool_param.has_pad_h() || pool_param.has_pad_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "pad_h & pad_w can only be used for 2D convolution.";
    CHECK_EQ(0, pool_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = pool_param.pad_h();
    pad_data[1] = pool_param.pad_w();
  } else {
    const int_tp num_pad_dims = pool_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
          num_pad_dims == num_spatial_axes_)
        << "pad must be specified once, or once per spatial dimension "
        << "(pad specified " << num_pad_dims << " times; "
        << num_spatial_axes_ << " spatial dims);";
    const int_tp kDefaultPad = 0;
    for (int_tp i = 0; i < num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
          pool_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }

  // Setup kernel stride dimensions
  dilation_.Reshape(spatial_dim_blob_shape);
  int_tp* dilation_data = dilation_.mutable_cpu_data();
  const int_tp num_dilation_dims = pool_param.dilation_size();
  CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
      num_dilation_dims == num_spatial_axes_)
      << "dilation must be specified once, or once per spatial dimension "
      << "(dilation specified " << num_dilation_dims << " times; "
      << num_spatial_axes_ << " spatial dims);";
  const int_tp kDefaultdilation = 1;
  for (int_tp i = 0; i < num_spatial_axes_; ++i) {
    dilation_data[i] =
        (num_dilation_dims == 0) ?
            kDefaultdilation :
            pool_param.dilation((num_dilation_dims == 1) ? 0 : i);
  }

  // Different 2D and ND im2col/col2im kernels for strided kernels
  use_skernel_ = false;
  for (int_tp i = 0; i < num_spatial_axes_; ++i) {
    use_skernel_ |= (dilation_data[i] != 1);
    if (use_skernel_) {
      break;
    }
  }

  this->InitializeQuantizers(bottom, top);
  Reshape(bottom, top);
}


template<typename Dtype, typename MItype, typename MOtype>
void PoolingLayer<Dtype, MItype, MOtype>::Reshape(
                                        const vector<Blob<MItype>*>& bottom,
                                        const vector<Blob<MOtype>*>& top) {
  vector<int_tp> size_shape(1, num_spatial_axes_);

  size_.Reshape(size_shape);
  pooled_size_.Reshape(size_shape);
  ext_kernel_shape_.Reshape(size_shape);
  int_tp* size_data = size_.mutable_cpu_data();
  int_tp* pooled_size_data = pooled_size_.mutable_cpu_data();
  int_tp* ext_kernel_shape_data = ext_kernel_shape_.mutable_cpu_data();
  int_tp* dilation_data = dilation_.mutable_cpu_data();
  int_tp* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  int_tp* pad_data = pad_.mutable_cpu_data();
  int_tp* stride_data = stride_.mutable_cpu_data();

  if (global_pooling_) {
    kernel_h_ = bottom[0]->height();
    kernel_w_ = bottom[0]->width();
  }
  switch (round_mode_) {
  case PoolingParameter_RoundMode_CEIL:
    pooled_height_ = static_cast<int>(ceil(static_cast<float>(
        height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
    pooled_width_ = static_cast<int>(ceil(static_cast<float>(
        width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
    break;
  case PoolingParameter_RoundMode_FLOOR:
    pooled_height_ = static_cast<int>(floor(static_cast<float>(
        height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
    pooled_width_ = static_cast<int>(floor(static_cast<float>(
        width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
    break;
  default:
    LOG(FATAL) << "Unknown rounding mode.";
  }
  if (pad_h_ || pad_w_) {
    // If we have padding, ensure that the last pooling starts strictly
    // inside the image (instead of at the padding); otherwise clip the last.
    if ((pooled_height_ - 1) * stride_h_ >= height_ + pad_h_) {
      --pooled_height_;
    }
  }

  vector<int_tp> top_shape = bottom[0]->shape();
  for (int_tp i = 0; i < num_spatial_axes_; ++i) {
    size_data[i] = bottom[0]->shape(channel_axis_ + 1 + i);
    ext_kernel_shape_data[i] = (kernel_shape_data[i] - 1) * dilation_data[i]
        + 1;
    pooled_size_data[i] = static_cast<int_tp>(ceil(
        static_cast<float>(size_data[i] + 2 * pad_data[i]
            - ext_kernel_shape_data[i]) / stride_data[i])) + 1;
    if (pad_data[i] > 0) {
      // If we have padding, ensure that the last pooling starts strictly
      // inside the image (instead of at the padding); otherwise clip the last.
      if ((pooled_size_data[i] - 1) * stride_data[i]
          >= size_data[i] + pad_data[i]) {
        --pooled_size_data[i];
      }
      CHECK_LT((pooled_size_data[i] - 1) * stride_data[i],
               size_data[i] + pad_data[i]);
    }
    top_shape[channel_axis_ + 1 + i] = pooled_size_data[i];
  }
  top[0]->Reshape(top_shape);
  if (top.size() > 1) {
    top[1]->ReshapeLike(*top[0]);
  }

  // If max pooling, we will initialize the vector index part.
  if (this->layer_param_.pooling_param().pool()
      == PoolingParameter_PoolMethod_MAX && top.size() == 1) {
    max_idx_.Reshape(top_shape);
  }

  // If stochastic pooling, we will initialize the random index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_STOCHASTIC) {
    rand_idx_.Reshape(top_shape);
  }

  if (Caffe::mode() == Caffe::GPU && this->device_program_.get() == nullptr) {
    this->GenerateProgram();
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void PoolingLayer<Dtype, MItype, MOtype>::Forward_cpu(
                                        const vector<Blob<MItype>*>& bottom,
                                        const vector<Blob<MOtype>*>& top) {
  int_tp kernel_h_ = kernel_shape_.cpu_data()[0];
  int_tp kernel_w_ = kernel_shape_.cpu_data()[1];
  int_tp stride_h_ = stride_.cpu_data()[0];
  int_tp stride_w_ = stride_.cpu_data()[1];
  int_tp pad_h_ = pad_.cpu_data()[0];
  int_tp pad_w_ = pad_.cpu_data()[1];
  int_tp height_ = size_.cpu_data()[0];
  int_tp width_ = size_.cpu_data()[1];
  int_tp pooled_height_ = pooled_size_.cpu_data()[0];
  int_tp pooled_width_ = pooled_size_.cpu_data()[1];

  const MItype* bottom_data = bottom[0]->cpu_data();
  MOtype* top_data = top[0]->mutable_cpu_data();
  const int_tp top_count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;  // suppress warnings about uninitialized variables
  Dtype* top_mask = NULL;
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.

  Dtype maxVal = FLT_MAX;
  if (std::is_same<MOtype, half_fp>::value) {
    maxVal = HALF_MAX;
  }
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // Initialize
    if (use_top_mask) {
      top_mask = top[1]->mutable_cpu_data();
      caffe_set(top_count, MOtype(-1), top_mask);
    } else {
      mask = max_idx_.mutable_cpu_data();
      caffe_set(top_count, (int_tp)-1, mask);
    }
    caffe_set(top_count, MOtype(-maxVal), top_data);
    // The main loop
    for (int_tp n = 0; n < bottom[0]->num(); ++n) {
      for (int_tp c = 0; c < channels_; ++c) {
        for (int_tp ph = 0; ph < pooled_height_; ++ph) {
          for (int_tp pw = 0; pw < pooled_width_; ++pw) {
            int_tp hstart = ph * stride_h_ - pad_h_;
            int_tp wstart = pw * stride_w_ - pad_w_;
            int_tp hend = min(hstart + kernel_h_, height_);
            int_tp wend = min(wstart + kernel_w_, width_);
            hstart = max(hstart, (int_tp)0);
            wstart = max(wstart, (int_tp)0);
            const int_tp pool_index = ph * pooled_width_ + pw;
            for (int_tp h = hstart; h < hend; ++h) {
              for (int_tp w = wstart; w < wend; ++w) {
                const int_tp index = h * width_ + w;
                if (bottom_data[index] > top_data[pool_index]) {
                  top_data[pool_index] = bottom_data[index];
                  if (use_top_mask) {
                    top_mask[pool_index] = static_cast<Dtype>(index);
                  } else {
                    mask[pool_index] = index;
                  }
                }
              }
            }
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += top[0]->offset(0, 1);
        } else {
          mask += top[0]->offset(0, 1);
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    for (int_tp i = 0; i < top_count; ++i) {
      top_data[i] = 0;
    }
    // The main loop
    for (int_tp n = 0; n < bottom[0]->num(); ++n) {
      for (int_tp c = 0; c < channels_; ++c) {
        for (int_tp ph = 0; ph < pooled_height_; ++ph) {
          for (int_tp pw = 0; pw < pooled_width_; ++pw) {
            int_tp hstart = ph * stride_h_ - pad_h_;
            int_tp wstart = pw * stride_w_ - pad_w_;
            int_tp hend = min(hstart + kernel_h_, height_ + pad_h_);
            int_tp wend = min(wstart + kernel_w_, width_ + pad_w_);
            int_tp pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, (int_tp)0);
            wstart = max(wstart, (int_tp)0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int_tp h = hstart; h < hend; ++h) {
              for (int_tp w = wstart; w < wend; ++w) {
                top_data[ph * pooled_width_ + pw] +=
                    bottom_data[h * width_ + w];
              }
            }
            top_data[ph * pooled_width_ + pw] /= pool_size;
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void PoolingLayer<Dtype, MItype, MOtype>::Backward_cpu(
                                        const vector<Blob<MOtype>*>& top,
                                        const vector<bool>& propagate_down,
                                        const vector<Blob<MItype>*>& bottom) {
  int_tp kernel_h_ = kernel_shape_.cpu_data()[0];
  int_tp kernel_w_ = kernel_shape_.cpu_data()[1];
  int_tp stride_h_ = stride_.cpu_data()[0];
  int_tp stride_w_ = stride_.cpu_data()[1];
  int_tp pad_h_ = pad_.cpu_data()[0];
  int_tp pad_w_ = pad_.cpu_data()[1];
  int_tp height_ = size_.cpu_data()[0];
  int_tp width_ = size_.cpu_data()[1];
  int_tp pooled_height_ = pooled_size_.cpu_data()[0];
  int_tp pooled_width_ = pooled_size_.cpu_data()[1];

  if (!propagate_down[0]) {
    return;
  }
  const MOtype* top_diff = top[0]->cpu_diff();
  MItype* bottom_diff = bottom[0]->mutable_cpu_diff();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int_tp* mask = NULL;  // suppress warnings about uninitialized variables
  const MOtype* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // The main loop
    if (use_top_mask) {
      top_mask = top[1]->cpu_data();
    } else {
      mask = max_idx_.cpu_data();
    }
    for (int_tp n = 0; n < top[0]->num(); ++n) {
      for (int_tp c = 0; c < channels_; ++c) {
        for (int_tp ph = 0; ph < pooled_height_; ++ph) {
          for (int_tp pw = 0; pw < pooled_width_; ++pw) {
            const int_tp index = ph * pooled_width_ + pw;
            const int_tp bottom_index =
                use_top_mask ? int_tp(top_mask[index]) : mask[index];
            bottom_diff[bottom_index] += top_diff[index];
          }
        }
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += top[0]->offset(0, 1);
        } else {
          mask += top[0]->offset(0, 1);
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    // The main loop
    for (int_tp n = 0; n < top[0]->num(); ++n) {
      for (int_tp c = 0; c < channels_; ++c) {
        for (int_tp ph = 0; ph < pooled_height_; ++ph) {
          for (int_tp pw = 0; pw < pooled_width_; ++pw) {
            int_tp hstart = ph * stride_h_ - pad_h_;
            int_tp wstart = pw * stride_w_ - pad_w_;
            int_tp hend = min(hstart + kernel_h_, height_ + pad_h_);
            int_tp wend = min(wstart + kernel_w_, width_ + pad_w_);
            int_tp pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, (int_tp)0);
            wstart = max(wstart, (int_tp)0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int_tp h = hstart; h < hend; ++h) {
              for (int_tp w = wstart; w < wend; ++w) {
                bottom_diff[h * width_ + w] +=
                  top_diff[ph * pooled_width_ + pw] / pool_size;
              }
            }
          }
        }
        // offset
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}

#ifdef CPU_ONLY
STUB_GPU(PoolingLayer);
#endif

INSTANTIATE_CLASS_3T_GUARDED(PoolingLayer, (half_fp), (half_fp),
                             PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(PoolingLayer, (float), (float),
                             PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(PoolingLayer, (double), (double),
                             PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(PoolingLayer, (uint8_t), (uint8_t),
                             PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(PoolingLayer, (uint16_t), (uint16_t),
                             PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(PoolingLayer, (uint32_t), (uint32_t),
                             PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(PoolingLayer, (uint64_t), (uint64_t),
                             PROTO_TYPES);

}    // namespace caffe
