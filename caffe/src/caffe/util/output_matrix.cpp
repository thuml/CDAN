#include "caffe/util/output_matrix.hpp"
#include "caffe/util/math_functions.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

namespace patch
{
    template < typename T > std::string to_string( const T& n )
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str() ;
    }
}

namespace caffe {

template <>
void write_to_file<float>(string filename, const int R, const int C, const float* A){
    std::ofstream output_file(filename.c_str(), std::ios::out);

    float* temp = new float[R * C];
    caffe_gpu_memcpy(R * C * sizeof(float), A, temp);
    
    for(int i = 0;i < R;++i){
        for(int j = 0;j < C - 1;++j){
            output_file << temp[i * C + j] << ",";
        }
        output_file << temp[i * C + C - 1] << "\r\n";
    }
    
    output_file.close();
    delete [] temp;
}

template <>
void write_to_file<double>(string filename, const int R, const int C, const double* A){
    std::ofstream output_file(filename.c_str(), std::ios::out);
    
    double* temp = new double[R * C];
    caffe_gpu_memcpy(R * C * sizeof(double), A, temp);
    
    for(int i = 0;i < R;++i){
        for(int j = 0;j < C - 1;++j){
            output_file << temp[i * C + j] << ",";
        }
        output_file << temp[i * C + C - 1] << "\r\n";
    }
    
    output_file.close();
    delete [] temp;
}

template <>
void print_gpu_matrix<float>(const float* M, int row, int col, int row_end, int col_end){
    int size = row * col;
    float *temp = new float[size];
    caffe_gpu_memcpy(size * sizeof(float), M, temp);
    string line;
    for(int i = 0;i < row_end;++i){
        line = "";
        for(int j = 0;j < col_end;++j){
            line += patch::to_string(temp[i * col + j]) + " ";
        }
        LOG(INFO) << line;
    }
    
    delete [] temp;
}

template <>
void print_gpu_matrix<double>(const double* M, int row, int col, int row_end, int col_end){
    int size = row * col;
    double *temp = new double[size];
    caffe_gpu_memcpy(size * sizeof(double), M, temp);
    string line;
    for(int i = 0;i < row_end;++i){
        line = "";
        for(int j = 0;j < col_end;++j){
            line += patch::to_string(temp[i * col + j]) + " ";
        }
        LOG(INFO) << line;
    }
    
    delete [] temp;
}

template <>
void print_gpu_matrix<float>(const float* M, int row, int col, int row_start, 
        int row_end, int col_start, int col_end){
    int size = row * col;
    float *temp = new float[size];
    caffe_gpu_memcpy(size * sizeof(float), M, temp);
    string line;
    for(int i = row_start;i < row_end;++i){
        line = "";
        for(int j = col_start;j < col_end;++j){
            line += patch::to_string(temp[i * col + j]) + " ";
        }
        LOG(INFO) << line;
    }
    
    delete [] temp;
}

template <>
void print_gpu_matrix<double>(const double* M, int row, int col, int row_start, 
        int row_end, int col_start, int col_end){
    int size = row * col;
    double *temp = new double[size];
    caffe_gpu_memcpy(size * sizeof(double), M, temp);
    string line;
    for(int i = row_start;i < row_end;++i){
        line = "";
        for(int j = col_start;j < col_end;++j){
            line += patch::to_string(temp[i * col + j]) + " ";
        }
        LOG(INFO) << line;
    }
    
    delete [] temp;
}

template <>
int check_nan_error<float>(const int n, const float* M){
    float *temp = new float[n];
    caffe_gpu_memcpy(n * sizeof(float), M, temp);
    for(int i = 0;i < n;++i){
        if(temp[i] != temp[i]){
            delete [] temp;
            return i;
        }
    }
    
    delete [] temp;
    return -1;
}

template <>
int check_nan_error<double>(const int n, const double* M){
    double *temp = new double[n];
    caffe_gpu_memcpy(n * sizeof(double), M, temp);
    for(int i = 0;i < n;++i){
        if(temp[i] != temp[i]){
            delete [] temp;
            return i;
        }
    }
    
    delete [] temp;
    return -1;
}

}
