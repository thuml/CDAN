/*

Copyright ©2016. The Regents of the University of California (Regents). All Rights Reserved. Permission to use, copy, modify, and distribute this software and its documentation for
educational, research, not-for-profit, and commercial purposes (such rights not subject to transfer), without fee, and without a signed licensing agreement, is hereby granted, provi
ded that the above copyright notice, this paragraph and the following two paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology Licensi
ng, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201, for commercial licensing opportunities.

Yang Gao, University of California, Berkeley.


IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE
 AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMP
ANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

*/

#include <vector>

#include "caffe/layers/bilinear_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void BilinearLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    // nothing should be done here
}

template<typename Dtype>
void BilinearLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    // check the shape of two bottom compatible
    for (int i = 0; i < 2; ++i) {
        const int num_axes = bottom[i]->num_axes();
        CHECK_EQ(num_axes, 4) << "Bilinear layer only support 4 dim blobs.";
    }
    for (int axis = 0; axis < 4; ++axis) {
        // the number of channels could be different
        if (axis == 1) {
            continue;
        }
        CHECK_EQ(bottom[0]->shape(axis), bottom[1]->shape(axis))
                << "Two bottom blobs not compatible at axis " << axis << ".";
    }

    // then assign the shape of the top blob
    vector<int> top_shape = bottom[0]->shape();
    top_shape[1] = bottom[0]->shape(1) * bottom[1]->shape(1);
    top_shape[2] = top_shape[3] = 1;
    top[0]->Reshape(top_shape);
}

template<typename Dtype>
void BilinearLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    Dtype* top_data = top[0]->mutable_cpu_data();
    const Dtype* bottom_data_0 = bottom[0]->cpu_data();
    const Dtype* bottom_data_1 = bottom[1]->cpu_data();

    const int step_top = top[0]->count(1);
    const int step_bottom_0 = bottom[0]->count(1);
    const int step_bottom_1 = bottom[1]->count(1);

    for (int b = 0; b < bottom[0]->shape(0); ++b) {
        // bottom0: C1*hw; bottom1: C2*hw;
        // will compute bottom0*bottom1'
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, bottom[0]->shape(1),
                bottom[1]->shape(1), bottom[0]->count(2), Dtype(1.0),
                bottom_data_0 + b * step_bottom_0,
                bottom_data_1 + b * step_bottom_1, Dtype(0.0),
                top_data + b * step_top);
    }
}

template<typename Dtype>
void BilinearLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down,
        const vector<Blob<Dtype>*>& bottom) {
    if ((!propagate_down[0]) && (!propagate_down[1]))
        return;
    // process the same bottom case
    // when the two bottoms are the same, one propagate down requires the other
    vector<bool> pd = propagate_down;
    if (bottom[0] == bottom[1])
        pd[0] = pd[1] = true;

    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff[2] = { bottom[0]->mutable_cpu_diff(), bottom[1]
            ->mutable_cpu_diff() };
    const Dtype* bottom_data[2] =
            { bottom[0]->cpu_data(), bottom[1]->cpu_data() };

    const int step_top = top[0]->count(1);
    const int step_bottom[2] = { bottom[0]->count(1), bottom[1]->count(1) };

    for (int b = 0; b < bottom[0]->shape(0); ++b) {
        for (int i = 0; i < 2; ++i)
            if (pd[i]) {
                // partial{L}/partial{bottomI} =
                // partial{L}/partial{y} * partial{y}/partial{bottomI} =
                // partial{L}/partial{y} * bottomJ
                // has to deal with the bottom[0]==bottom[1] case
                if ((bottom[0] == bottom[1]) && (i == 1)) {
                    // then multiply the output by 2
                    caffe_scal<Dtype>(step_bottom[0], Dtype(2.0),
                            bottom_diff[0] + b * step_bottom[0]);
                    continue;
                }
                // top shape: C1*C2; For i==0, second is C2*hw, output is C1*hw
                CBLAS_TRANSPOSE TransTop =
                        ((i == 0) ? CblasNoTrans : CblasTrans);
                caffe_cpu_gemm<Dtype>(TransTop, CblasNoTrans,
                        bottom[i]->shape(1), bottom[0]->count(2),
                        bottom[1 - i]->shape(1), Dtype(1.0),
                        top_diff + b * step_top,
                        bottom_data[1 - i] + b * step_bottom[1 - i], Dtype(0.0),
                        bottom_diff[i] + b * step_bottom[i]);
            }
    }
}

#ifdef CPU_ONLY
STUB_GPU(BilinearLayer);
#endif

INSTANTIATE_CLASS(BilinearLayer);
REGISTER_LAYER_CLASS(Bilinear);

}  // namespace caffe
