#ifndef CAFFE_RANDOM_MACLAURIN_LAYER_HPP_
#define CAFFE_RANDOM_MACLAURIN_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"


namespace caffe {

template <typename Dtype>
class RandomMaclaurinLayer : public Layer<Dtype> {
 public:
  explicit RandomMaclaurinLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "RandomMaclaurin"; }
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
 protected:
  Dtype loss_weight_;
  Blob<Dtype> W1_, W2_;
  Blob<Dtype> projection_, projection_diff_;
  int* M1_,*M2_;
  int num_output_;
  Dtype sqrt_output_;
  int epoch_progress_, epoch_size_;
  string distribution_type_;
  Dtype normalize_factor_, std1_, std2_, a1_, a2_, b1_, b2_;
};

}

#endif

