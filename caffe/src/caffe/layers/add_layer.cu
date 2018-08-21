#include <algorithm>
#include <vector>

#include "caffe/layers/add_layer.hpp"
#include "caffe/util/output_matrix.hpp"

namespace caffe {

template <typename Dtype>
__global__ void AddForwardGPU2(const int nthreads, const Dtype* bottom1, const Dtype* bottom2, const Dtype weight1, const Dtype weight2, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
     top_data[index] = weight1 * bottom1[index] + weight2 * bottom2[index];
  }
}

template <typename Dtype>
__global__ void AddForwardGPU3(const int nthreads, const Dtype* bottom1, const Dtype* bottom2, const Dtype* bottom3, const Dtype weight1, const Dtype weight2, const Dtype weight3, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
     top_data[index] = weight1 * bottom1[index] + weight2 * bottom2[index] + weight3 * bottom3[index];
  }
}


template <typename Dtype>
void AddLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  if(direct_add_){
    //LOG(INFO) << "fc data";
    //print_gpu_matrix(bottom[0]->gpu_data(), bottom[0]->shape(0), bottom[0]->count(1), 0, 1, 0, bottom[0]->count(1));
    //LOG(INFO) << "shortcut data";
    //print_gpu_matrix(bottom[1]->gpu_data(), bottom[1]->shape(0), bottom[1]->count(1), 0, 1, 0, bottom[1]->count(1));
    caffe_gpu_add(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), top[0]->mutable_gpu_data());
    return;
  }
  else if(bottom_number_ == 2){
    LOG(INFO) << "fc data";
    print_gpu_matrix(bottom[0]->gpu_data(), bottom[0]->shape(0), bottom[0]->count(1), 0, 1, 0, bottom[0]->count(1));
    LOG(INFO) << "shortcut data";
    print_gpu_matrix(bottom[1]->gpu_data(), bottom[1]->shape(0), bottom[1]->count(1), 0, 1, 0, bottom[1]->count(1));
    AddForwardGPU2<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), contribution_weight_[0], contribution_weight_[1], top[0]->mutable_gpu_data()); 
  }
  else if(bottom_number_ == 3){
    AddForwardGPU3<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), bottom[2]->gpu_data(), contribution_weight_[0], contribution_weight_[1], contribution_weight_[2], top[0]->mutable_gpu_data()); }
}

template <typename Dtype>
__global__ void AddBackwardGPU2(const int nthreads, Dtype* bottom1, Dtype* bottom2, const Dtype weight1, const Dtype weight2, const Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
     bottom1[index] = top_data[index] * weight1;
     bottom2[index] = weight2 * top_data[index];
  }
}

template <typename Dtype>
__global__ void AddBackwardGPU3(const int nthreads, Dtype* bottom1, Dtype* bottom2, Dtype* bottom3, const Dtype weight1, const Dtype weight2, const Dtype weight3,  const Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
     bottom1[index] = top_data[index] * weight1;
     bottom2[index] = weight2 * top_data[index];
     bottom3[index] = weight3 * top_data[index];
  }
}

template <typename Dtype>
void AddLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  int count = bottom[0]->count();
  if (propagate_down[0]) {
    if (direct_add_){
      caffe_gpu_memcpy(count * sizeof(Dtype), top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff());
      caffe_gpu_memcpy(count * sizeof(Dtype), top[0]->gpu_diff(), bottom[1]->mutable_gpu_diff());
      return;
    }
    else if(bottom_number_ == 2){
      AddBackwardGPU2<Dtype><<<CAFFE_GET_BLOCKS(count),
          CAFFE_CUDA_NUM_THREADS>>>(count, bottom[0]->mutable_gpu_diff(), bottom[1]->mutable_gpu_diff(), contribution_weight_[0], contribution_weight_[1], top[0]->gpu_diff()); 
      if(weight_rate_ != 0 ){
        Dtype grad1, grad2;
        caffe_gpu_dot(count, top[0]->gpu_diff(), bottom[0]->gpu_data(), &grad1);
        caffe_gpu_dot(count, top[0]->gpu_diff(), bottom[1]->gpu_data(), &grad2);
        contribution_weight_[0] += weight_rate_ * grad1;
        contribution_weight_[1] += weight_rate_ * grad2;  
        if(contribution_weight_[0] < 0){
          contribution_weight_[0] = 0;
        }
        if(contribution_weight_[1] < 0){
          contribution_weight_[1] = 0;
        }
      }
      LOG(INFO) << "weight1: " << contribution_weight_[0] << " weight2: " << contribution_weight_[1];
    }
    else if(bottom_number_ == 3){
      AddBackwardGPU3<Dtype><<<CAFFE_GET_BLOCKS(count),
          CAFFE_CUDA_NUM_THREADS>>>(count, bottom[0]->mutable_gpu_diff(), bottom[1]->mutable_gpu_diff(),bottom[2]->mutable_gpu_diff(), contribution_weight_[0], contribution_weight_[1], contribution_weight_[2], top[0]->gpu_diff()); 
      if(weight_rate_ != 0 ){
        Dtype grad1, grad2, grad3;
        caffe_gpu_dot(count, top[0]->gpu_diff(), bottom[0]->gpu_data(), &grad1);
        caffe_gpu_dot(count, top[0]->gpu_diff(), bottom[1]->gpu_data(), &grad2);
        caffe_gpu_dot(count, top[0]->gpu_diff(), bottom[2]->gpu_data(), &grad3);
        contribution_weight_[0] += weight_rate_ * grad1;
        contribution_weight_[1] += weight_rate_ * grad2;  
        contribution_weight_[2] += weight_rate_ * grad3;  
        if(contribution_weight_[0] < 0){
          contribution_weight_[0] = 0;
        }
        if(contribution_weight_[1] < 0){
          contribution_weight_[1] = 0;
        }
        if(contribution_weight_[2] < 0){
          contribution_weight_[2] = 0;
        }
      }
      LOG(INFO) << "weight1: " << contribution_weight_[0] << " weight2: " << contribution_weight_[1] << " weight3: " << contribution_weight_[2];
    }
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(AddLayer);


}  // namespace caffe
