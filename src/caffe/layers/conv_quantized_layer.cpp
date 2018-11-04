#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/conv_quantized_layer.hpp"
#include "dkm/include/dkm.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionQuantizedLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void ConvolutionQuantizedLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (this->mask_term_) {
    weights_quantized_shape_.clear();
    weights_quantized_shape_.push_back(this->blobs_[2]->count());
    weights_quantized_.Reshape(weights_quantized_shape_);
    output_saliencies_.Reshape(this->output_shape_);
    centroids_shape_.clear();
    centroids_shape_.push_back(this->layer_param_.convolution_quantized_param().centroids());
    centroids_.Reshape(centroids_shape_);
  }
  BaseConvolutionLayer<Dtype>::Reshape(bottom, top);
}

struct IsPresent {
    IsPresent(float value) : value(value) {}
    bool operator()(const std::array<float, 1>& arr ) {
        auto v = std::find(std::begin(arr), std::end(arr), value);
        return v != std::end(arr);
    }
private:
    float value;
};

// bool IsPresent(const std::array<float, 1>& arr, double value)
// {
//   auto v = std::find(std::begin(arr), std::end(arr), value);
//   return v != std::end(arr)
// }

template <typename Dtype>
void ConvolutionQuantizedLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_quantized = this->weights_quantized_.mutable_cpu_data();
  const int count = this->blobs_[0]->count();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    switch (this->layer_param_.convolution_quantized_param().method()) {
      case (0): {
        unsigned no_centroids = this->layer_param_.convolution_quantized_param().centroids();
        const Dtype* centroids = this->centroids_.cpu_data();
        (void)no_centroids;
        (void)centroids;
        //TODO: call kmeans here! results in weight_quantized, and delete the line below


        // where first is centroids, second is clusters

        // for (auto i = 0; i != count; i++){
        //   auto v = weight_quantized[i];
        //   auto it = std::find_if (std::get<1>(clusters).begin(), std::get<1>(clusters).end(), IsPresent(v));
        //   weight_quantized[i] = *it;
        // }

        caffe_copy(count, weight, weight_quantized);
      } break;

      case(1): {
        std::bitset<8*sizeof(Dtype)> mantissa_mask;
        mantissa_mask.flip();
        mantissa_mask <<= this->layer_param_.convolution_quantized_param().truncate_bits();
        caffe_and(count, mantissa_mask, weight, weight_quantized);
      } break;

      default: {
        caffe_copy(count, weight, weight_quantized);
      } break;
    }

    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight_quantized,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}


template <typename Dtype>
void map_data(std::vector<std::array<float, 1>> centroids, caffe::Blob<Dtype>* data){
  for( auto i : centroids){
    // std::cout << i[0] << std::endl;
  }
}

template <typename Dtype>
void ConvolutionQuantizedLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
    // Compute saliency for each output
    int outputs = 1;
    for (int i = 0; i < this->output_shape_.size(); i++) {
      outputs *= this->output_shape_[i];
    }

    Dtype* saliency_data = this->output_saliencies_.mutable_cpu_data();
    caffe_mul(outputs, bottom_data, bottom_diff, saliency_data);
    caffe_powx(outputs, saliency_data, (Dtype)2, saliency_data);

    Dtype* centroids = this->centroids_.mutable_cpu_data();
    (void)centroids;


    // TODO: update the centroids here with the saliency data
    Dtype* wq = this->weights_quantized_.mutable_cpu_data();

    const int count = this->weights_quantized_.count();

    std::vector<std::array<float, 1>> values;


    for(auto i = 0; i != count; i++){
      std::array<float, 1> tmp;
      tmp = this->weights_quantized_.data()[i];
      values.push_back(tmp);
    }
    // std::vector<std::array<Dtype, 1>> initial_data(wq, wq + count);
    // auto clusters = dkm::kmeans_lloyd(initial_data, 16);  // Returns (Clusters, Data)
    // auto means = std::get<0>(clusters);
    // map_data(means, &this->weights_quantized_);



    // caffe_copy(16, (Dtype *)(means), this->centroids_);

  }
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionQuantizedLayer);
#endif

INSTANTIATE_CLASS(ConvolutionQuantizedLayer);
//REGISTER_LAYER_CLASS(ConvolutionQuantized);

}  // namespace caffe
