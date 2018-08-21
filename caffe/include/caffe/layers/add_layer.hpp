#ifndef CAFFE_ADD_LAYER_HPP_
#define CAFFE_ADD_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

template <typename Dtype>
class AddLayer : public NeuronLayer<Dtype> {
 public:
  explicit AddLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  ~AddLayer(){delete contribution_weight_;}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "Add"; }
virtual inline int ExactNumBottomBlobs() const { return -1; }


 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  int num_;
  int bottom_number_;
  Dtype* contribution_weight_;
  Dtype weight_rate_;
  bool direct_add_;
};

}  // namespace caffe

#endif  // CAFFE_TANH_LAYER_HPP_
