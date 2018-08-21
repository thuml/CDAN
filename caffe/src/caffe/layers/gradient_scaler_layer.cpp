#include <algorithm>
#include <vector>

#include "caffe/layers/gradient_scaler_layer.hpp"
#include "caffe/messenger.hpp"

namespace caffe {

class AdaptationCoefficientHandler: public Listener {
 public:
  AdaptationCoefficientHandler(float lower_bound, float upper_bound, 
                               float alpha, float max_iter, float* coeff, int threshold_iter)
      : lower_bound_(lower_bound), upper_bound_(upper_bound), alpha_(alpha),
        max_iter_(max_iter), coeff_(*coeff) {
    height_ = upper_bound_ - lower_bound_;
    threshold_iter1_ = threshold_iter;
  }

  void handle(void* message) {
    int iter = *(static_cast<int*>(message));
    if(iter > threshold_iter1_){
      float progress = std::min(1.f, static_cast<float>(iter-threshold_iter1_) / max_iter_);
      coeff_ = 2.f * height_ / (1.f + exp(-alpha_ * progress)) - 
             height_ + lower_bound_;
    }
    else{
      coeff_ = 0;
    }

    // LOG(INFO) << "iter = " << iter << " progress = " << progress << " coeff = " << coeff_;
  }

 private:
  float lower_bound_, upper_bound_, alpha_, max_iter_, height_, threshold_iter1_;
  float& coeff_;
};

template <typename Dtype>
void GradientScalerLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);

  lower_bound_ = this->layer_param_.gradient_scaler_param().lower_bound();
  upper_bound_ = this->layer_param_.gradient_scaler_param().upper_bound();
  threshold_iter_ = this->layer_param_.gradient_scaler_param().threshold_iter();
  iter_ = 0;
  alpha_ = this->layer_param_.gradient_scaler_param().alpha();
  max_iter_ = threshold_iter_ + this->layer_param_.gradient_scaler_param().max_iter();
  coeff_ = 1.f; // Default adaptation coefficient.

  DCHECK(lower_bound_ <= upper_bound_);
  DCHECK(alpha_ >= 0.f);
  DCHECK(max_iter_ >= 1.f);
  
  Messenger::AddListener("SOLVER_ITER_CHANGED", 
      new AdaptationCoefficientHandler(lower_bound_, upper_bound_, 
                                       alpha_, max_iter_, &coeff_, threshold_iter_));
}

template <typename Dtype>
void GradientScalerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ShareData(*bottom[0]);
}

template <typename Dtype>
void GradientScalerLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const int count = bottom[0]->count();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

    caffe_cpu_scale(count, Dtype(-coeff_), top_diff, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(GradientScalerLayer);
#endif

INSTANTIATE_CLASS(GradientScalerLayer);
REGISTER_LAYER_CLASS(GradientScaler);

}  // namespace caffe
