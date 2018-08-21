#include <vector>

#include "caffe/layers/add_layer.hpp"

namespace caffe {

template <typename Dtype>
void AddLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = bottom.size();
  bottom_number_ = bottom.size();
  contribution_weight_ = new Dtype[bottom_number_];
  weight_rate_ = this->layer_param_.adding_param().weight_rate();
  direct_add_ = this->layer_param_.adding_param().direct_add();
  if (bottom_number_ == 2){
    contribution_weight_[0] = this->layer_param_.adding_param().weight1();
    contribution_weight_[1] = 1 - contribution_weight_[0];
  }
  else if(bottom_number_ == 3){
    contribution_weight_[0] = this->layer_param_.adding_param().weight1();
    contribution_weight_[1] = this->layer_param_.adding_param().weight2();
    contribution_weight_[2] = 1 - contribution_weight_[0] - contribution_weight_[1];
  }
}

template <typename Dtype>
void AddLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*(bottom[0]));
  caffe_gpu_set(top[0]->count(), Dtype(0), top[0]->mutable_gpu_data());
}

template <typename Dtype>
void AddLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void AddLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(AddLayer);
#endif

INSTANTIATE_CLASS(AddLayer);
REGISTER_LAYER_CLASS(Add);

}  // namespace caffe
