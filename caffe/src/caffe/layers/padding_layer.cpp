#include <vector>

#include "caffe/layers/padding_layer.hpp"

namespace caffe {

template <typename Dtype>
void PaddingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int bottom_size = bottom.size();
  const PaddingParameter& param = this->layer_param_.padding_param();
  CHECK_EQ(bottom_size, param.all_pad_size()) << "The number of padding shapes should match the number of bottom layers.";
  CHECK_EQ(bottom_size, top.size()) << "Bottom and top blobs should match.";
  for(int i = 0; i < bottom_size; i++){
    padding_size_.push_back(vector<int>());
    int temp_size = param.all_pad(i).pad_size();
    for(int j = 0; j < temp_size; j ++){
      padding_size_[i].push_back(param.all_pad(i).pad(j));
    }
  }
  for(int i = 0; i < bottom_size; i ++){
    top[i]->Reshape(padding_size_[i]);
  }
}

template <typename Dtype>
void PaddingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void PaddingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void PaddingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(PaddingLayer);
#endif

INSTANTIATE_CLASS(PaddingLayer);
REGISTER_LAYER_CLASS(Padding);

}  // namespace caffe
