#include <algorithm>
#include <vector>

#include <caffe/layers/gradient_scaler_layer.hpp>

namespace caffe {

template <typename Dtype>
void GradientScalerLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    iter_ += 1;
  top[0]->ShareData(*bottom[0]);
}

template <typename Dtype>
void GradientScalerLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const int count = top[0]->count();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    if(iter_ > threshold_iter_){
        caffe_gpu_scale(count, Dtype(-coeff_), top_diff, bottom_diff);
    }
    else{
        caffe_gpu_scale(count, Dtype(0), top_diff, bottom_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(GradientScalerLayer);

}  // namespace caffe
