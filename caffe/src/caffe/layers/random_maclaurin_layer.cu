#include <vector>

#include "caffe/layers/random_maclaurin_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/output_matrix.hpp"

namespace caffe {

template <typename Dtype>
__global__ void RandomMaclaurinForward(const int nthreads, const Dtype* data, Dtype* data1) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    if(data[index] == 0){
      data1[index] = Dtype(-1);
    }    
  }
}

template <typename Dtype>
void RandomMaclaurinLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    const int nthreads = bottom[0]->shape(0);
    const int axis1 = bottom[0]->shape(1);
    const int axis2 = bottom[1]->shape(1);

    caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, nthreads, num_output_, axis1, Dtype(1), 
        bottom[0]->gpu_data(), this->blobs_[0]->gpu_data(), Dtype(0), projection_.mutable_gpu_data());
    caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, nthreads, num_output_, axis2, Dtype(1), 
        bottom[1]->gpu_data(), this->blobs_[1]->gpu_data(), Dtype(0), projection_.mutable_gpu_diff());
    caffe_gpu_mul(nthreads * num_output_, projection_.gpu_data(), projection_.gpu_diff(), 
        top[0]->mutable_gpu_data());
    caffe_gpu_scal(nthreads * num_output_, Dtype(1.0 / normalize_factor_), top[0]->mutable_gpu_data());
}
/*
template <typename Dtype>
__global__ void RandomMaclaurinBackward(const int nthreads, const Dtype* data, const Dtype* data1, const Dtype* data2,
    const int axis1, const int axis2, Dtype* diff1, Dtype* diff2) {
  CUDA_KERNEL_LOOP(index, nthreads) {
      int temp = axis1 * axis2;
      for(int i = 0; i < axis1; i++){
           diff1[index * axis1 + i] = 0;
      }
    for(int j = 0; j < axis2; j++){
           diff2[index * axis2 + j] = 0;
      }     
     for(int i = 0; i < axis1; i++){
         for(int j = 0; j < axis2; j++){
             diff1[index * axis1 + i] += data[index * temp + i * axis2 + j] * data2[index * axis2 + j];
         }
         //diff1[index * axis1 + i] /= Dtype(axis2);
     }
     for(int j = 0; j < axis2; j++){
         for(int i = 0; i < axis1; i++){
             diff2[index * axis2 + j] += data[index * temp + i * axis2 + j] * data1[index * axis1 + i];
         }
         //diff2[index * axis2 + j] /= Dtype(axis1);
     }
  }
}*/

template <typename Dtype>
void RandomMaclaurinLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    const int nthreads = bottom[0]->shape(0);
    const int axis1 = bottom[0]->shape(1);
    const int axis2 = bottom[1]->shape(1);
    caffe_gpu_mul(nthreads * num_output_, top[0]->gpu_diff(), projection_.gpu_diff(), projection_diff_.mutable_gpu_data());
    caffe_gpu_mul(nthreads * num_output_, top[0]->gpu_diff(), projection_.gpu_data(), projection_diff_.mutable_gpu_diff());
    caffe_gpu_gemm(CblasNoTrans, CblasTrans, nthreads, axis1, num_output_, Dtype(1), projection_diff_.gpu_data(), this->blobs_[0]->gpu_data(), Dtype(0), bottom[0]->mutable_gpu_diff()); 
    caffe_gpu_gemm(CblasNoTrans, CblasTrans, nthreads, axis2, num_output_, Dtype(1), projection_diff_.gpu_diff(), this->blobs_[1]->gpu_data(), Dtype(0), bottom[1]->mutable_gpu_diff()); 
    caffe_gpu_scal(bottom[0]->count(), Dtype(0.0 / normalize_factor_), bottom[0]->mutable_gpu_diff());
    caffe_gpu_scal(bottom[1]->count(), Dtype(1.0 / normalize_factor_), bottom[1]->mutable_gpu_diff());
    caffe_gpu_set(this->blobs_[0]->count(), Dtype(0.0), this->blobs_[0]->mutable_gpu_diff());
    caffe_gpu_set(this->blobs_[1]->count(), Dtype(0.0), this->blobs_[1]->mutable_gpu_diff());
    //RandomMaclaurinBackward<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
    //  nthreads, top[0]->gpu_diff(), bottom[0]->gpu_data(), bottom[1]->gpu_data(), axis1, axis2, bottom[0]->mutable_gpu_diff(), bottom[1]->mutable_gpu_diff());
}
INSTANTIATE_LAYER_GPU_FUNCS(RandomMaclaurinLayer);

}  // namespace caffe
