#include <vector>
#include <algorithm>
#include <cfloat>

#include "caffe/layers/sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void FocalParameter(const int nthreads, const Dtype* input_data, const Dtype* target, Dtype* focal_parameter){
  CUDA_KERNEL_LOOP(index, nthreads){
    if(target[index] == 0){
        focal_parameter[index] = pow(Dtype(1.0-input_data[index]), Dtype(0.1));
    }
    else{
        focal_parameter[index] = pow(Dtype(input_data[index]), Dtype(0.1));
    }
  }
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Backward_gpu(
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
    const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
    const Dtype* target = bottom[1]->gpu_data();

    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_copy(count, sigmoid_output_data, bottom_diff);
    caffe_gpu_axpy(count, Dtype(-1), target, bottom_diff);
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_gpu_scal(count, loss_weight / num, bottom_diff);


    // add focal parameter
    if (use_focal_){
        Dtype* focal_parameter = focal_parameter_.mutable_gpu_data();
        FocalParameter<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, sigmoid_output_data, target, focal_parameter);
        Dtype all_focal;
        caffe_gpu_asum(count, focal_parameter, &all_focal);
        ave_focal_ = all_focal / Dtype(count);
        caffe_gpu_mul(count, focal_parameter, bottom_diff, bottom_diff);
        caffe_gpu_scal(count, Dtype(1. / ave_focal_), bottom_diff);
    }   
  }
}

INSTANTIATE_LAYER_GPU_BACKWARD(SigmoidCrossEntropyLossLayer);


}  // namespace caffe
