#include <vector>

#include "caffe/layers/cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void CrossEntropyBackward(const int nthreads, const Dtype* bottom_data, const Dtype* bottom_label, Dtype* bottom_diff){
  CUDA_KERNEL_LOOP(index, nthreads) {
    if (bottom_data[index] > 0 && bottom_data[index] < 1){
      bottom_diff[index] = (bottom_data[index] - bottom_label[index]) / ((1 - bottom_data[index]) * bottom_data[index]);
    }
    else{
      bottom_diff[index] = 0;
    }
  }
}

template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* target = bottom[1]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    CrossEntropyBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, bottom_data, target, bottom_diff);
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_gpu_scal(count, loss_weight / num, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_BACKWARD(CrossEntropyLossLayer);


}  // namespace caffe
