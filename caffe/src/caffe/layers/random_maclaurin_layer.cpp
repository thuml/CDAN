#include <algorithm>
#include <vector>

#include "caffe/layers/random_maclaurin_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RandomMaclaurinLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    num_output_ = this->layer_param_.random_maclaurin_param().num_output();
    distribution_type_ = this->layer_param_.random_maclaurin_param().distribution_type();
    vector<int> temp_shape;
    temp_shape.push_back(bottom[0]->shape(0));
    temp_shape.push_back(num_output_);
    projection_.Reshape(temp_shape);
    projection_diff_.Reshape(temp_shape);
    top[0]->Reshape(temp_shape);
    temp_shape[1] = num_output_;
    temp_shape[0] = bottom[0]->count(1);
    this->blobs_.resize(2);
    this->blobs_[0].reset(new Blob<Dtype>(temp_shape));
    //W1_.Reshape(temp_shape);
    temp_shape[0] = bottom[1]->count(1);
    this->blobs_[1].reset(new Blob<Dtype>(temp_shape));
    //W2_.Reshape(temp_shape);

    //select random matrix distribution
    int count1 = this->blobs_[0]->count();
    int count2 = this->blobs_[1]->count();
    if (distribution_type_ == "bernoulli"){
      M1_ = new int[count1];
      M2_ = new int[count2];
      normalize_factor_ = num_output_;
    }
    else if (distribution_type_ == "gaussian"){
      M1_ = NULL;
      M2_ = NULL;
      std1_ = this->layer_param_.random_maclaurin_param().std1();
      std2_ = this->layer_param_.random_maclaurin_param().std2();
      normalize_factor_ = num_output_ * std1_ * std2_;
    }
    else if (distribution_type_ == "uniform"){
      M1_ = NULL;
      M2_ = NULL;
      a1_ = this->layer_param_.random_maclaurin_param().a1();
      b1_ = this->layer_param_.random_maclaurin_param().b1();
      a2_ = this->layer_param_.random_maclaurin_param().a2();
      b2_ = this->layer_param_.random_maclaurin_param().b2();
      normalize_factor_ = num_output_ * b1_ * b2_ / Dtype(3);
    }

    epoch_size_ = this->layer_param_.random_maclaurin_param().epoch_size();
    epoch_progress_ = epoch_size_ + 1;
}

template <typename Dtype>
void RandomMaclaurinLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    vector<int> temp_shape;
    temp_shape.push_back(bottom[0]->shape(0));
    temp_shape.push_back(num_output_);
    top[0]->Reshape(temp_shape);
    int count1 = this->blobs_[0]->count();
    int count2 = this->blobs_[1]->count();

    if (epoch_progress_ > epoch_size_){
      epoch_progress_ = 0;
      if (distribution_type_ == "bernoulli"){
        Dtype* W1_cpu = this->blobs_[0]->mutable_cpu_data();
        Dtype* W2_cpu = this->blobs_[1]->mutable_cpu_data();
        caffe_rng_bernoulli(count1, 0.5, M1_);
        caffe_rng_bernoulli(count2, 0.5, M2_);
        for(int i = 0; i < count1; i++){
          W1_cpu[i] = Dtype(M1_[i]);
        }
        for(int i = 0; i < count2; i++){
          W2_cpu[i] = Dtype(M2_[i]);
        }
        caffe_gpu_memcpy(count1 * sizeof(Dtype), W1_cpu, this->blobs_[0]->mutable_gpu_data());
        caffe_gpu_memcpy(count2 * sizeof(Dtype), W2_cpu, this->blobs_[1]->mutable_gpu_data());
      }
      else if (distribution_type_ == "gaussian"){
        caffe_gpu_rng_gaussian<Dtype>(count1, Dtype(0), std1_, this->blobs_[0]->mutable_gpu_data());
        caffe_gpu_rng_gaussian<Dtype>(count2, Dtype(0), std2_, this->blobs_[1]->mutable_gpu_data());
      }
      else if (distribution_type_ == "uniform"){
        caffe_gpu_rng_uniform<Dtype>(count1, a1_, b1_, this->blobs_[0]->mutable_gpu_data());
        caffe_gpu_rng_uniform<Dtype>(count2, a2_, b2_, this->blobs_[1]->mutable_gpu_data());
       
      }
    }
    epoch_progress_ += 1;
}

template <typename Dtype>
void RandomMaclaurinLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void RandomMaclaurinLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(RandomMaclaurinLayer);
#endif

INSTANTIATE_CLASS(RandomMaclaurinLayer);
REGISTER_LAYER_CLASS(RandomMaclaurin);

}  // namespace caffe
