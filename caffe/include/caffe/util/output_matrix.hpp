#include "glog/logging.h"
#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"

namespace caffe{

template <typename Dtype>
void write_to_file(string filename, const int R, const int C, const Dtype* A);

template <typename Dtype>
void print_gpu_matrix(const Dtype* M, int row, int col, int row_end, int col_end);

template <typename Dtype>
void print_gpu_matrix(const Dtype* M, int row, int col, int row_start, 
        int row_end, int col_start, int col_end);

template <typename Dtype>
int check_nan_error(const int n, const Dtype* M);

}
