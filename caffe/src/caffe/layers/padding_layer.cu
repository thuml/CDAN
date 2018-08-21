#include <algorithm>
#include <vector>

#include "caffe/layers/padding_layer.hpp"

namespace caffe {


template <typename Dtype>
__global__ void PaddingCopyData(const int nthreads, const int dim_source, const int dim_target, const Dtype* source, Dtype* target){
  CUDA_KERNEL_LOOP(index, nthreads){
    int index_data = index / dim_target;
    int pos_data = index % dim_target;
    if(pos_data < dim_source){
      target[index] = source[index_data * dim_source + pos_data];
    }
    else{
      target[index] = 0;
    }
  }
}

template <typename Dtype>
void PaddingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int num_blobs = bottom.size();
  int count;
  for(int i = 0; i < num_blobs; i++){
    count = top[i]->count();
    PaddingCopyData<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(count, bottom[i]->count(1), top[i]->count(1), bottom[i]->gpu_data(), top[i]->mutable_gpu_data());
  }
}

template <typename Dtype>
__global__ void PaddingCopyDiff(const int nthreads, const int dim_source, const int dim_target, Dtype* source, const Dtype* target){
  CUDA_KERNEL_LOOP(index, nthreads){
    int index_data = index / dim_target;
    int pos_data = index % dim_target;
    if(pos_data < dim_source){
      source[index_data * dim_source + pos_data] = target[index];
    }
  }
}


template <typename Dtype>
void PaddingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    int num_blobs = bottom.size();
    int count;
    for(int i = 0; i < num_blobs; i++){
      count = top[i]->count();
      PaddingCopyDiff<Dtype><<<CAFFE_GET_BLOCKS(count),
          CAFFE_CUDA_NUM_THREADS>>>(count, bottom[i]->count(1), top[i]->count(1), bottom[i]->mutable_gpu_diff(), top[i]->gpu_diff());
    }     
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(PaddingLayer);


}  // namespace caffe
