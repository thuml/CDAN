#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/aggregate_weight_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void AggregateWeightForward(const int nthreads, const int dim, const Dtype* source, Dtype* target) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    int out_place = i % dim;
    target[out_place] += source[i];
  }
}

template <typename Dtype>
void AggregateWeightLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* temp_weight = new Dtype[num_output_];
  for (int i = 0; i < num_output_; i++){
    temp_weight[i] = 0;
  }
  AggregateWeightForward<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(bottom[0]->count(), num_output_, bottom[0]->gpu_data(), temp_weight);
  for(int i = 0; i < num_output_; i++){
    caffe_gpu_memcpy(sizeof(Dtype), &temp_weight[i], top[i]->mutable_gpu_data());
  }
  delete [] temp_weight;
}

template <typename Dtype>
void AggregateWeightLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {

  }
}

INSTANTIATE_LAYER_GPU_FUNCS(AggregateWeightLayer);

}  // namespace caffe
