#include <vector>

#include "caffe/layers/multiply_weight_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/output_matrix.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MultiplyWeightForward(const int nthreads, const Dtype* data1, const Dtype* data2,
    const int axis1, Dtype* out_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int data_id = nthreads / axis1;
    out_data[index] = data1[index] * data2[data_id];  
  }
}

template <typename Dtype>
void MultiplyWeightLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    const int nthreads = bottom[0]->count();
    const int axis1 = bottom[0]->shape(1);
    MultiplyWeightForward<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, bottom[0]->gpu_data(), bottom[1]->gpu_data(), axis1, top[0]->mutable_gpu_data());
}

template <typename Dtype>
__global__ void MultiplyWeightBackward(const int nthreads, const Dtype* diff1, const Dtype* data2,
    const int axis1, Dtype* diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int data_id = nthreads / axis1;
    diff[index] = diff1[index] * data2[data_id];
  }
}

template <typename Dtype>
void MultiplyWeightLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    const int nthreads = bottom[0]->count();
    const int axis1 = bottom[0]->shape(1); 
    MultiplyWeightBackward<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, top[0]->gpu_diff(), bottom[1]->gpu_data(), axis1, bottom[0]->mutable_gpu_diff());
    caffe_gpu_scal(bottom[1]->count(), Dtype(0), bottom[1]->mutable_gpu_diff());
}
INSTANTIATE_LAYER_GPU_FUNCS(MultiplyWeightLayer);

}  // namespace caffe
