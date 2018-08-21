#include <algorithm>
#include <vector>

#include "caffe/layers/multiply_weight_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MultiplyWeightLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

}

template <typename Dtype>
void MultiplyWeightLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void MultiplyWeightLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void MultiplyWeightLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(MultiplyWeightLayer);
#endif

INSTANTIATE_CLASS(MultiplyWeightLayer);
REGISTER_LAYER_CLASS(MultiplyWeight);

}  // namespace caffe
