# CDAN implemented in Caffe

## Prerequisites
- [Caffe](http://caffe.berkeleyvision.org/) 
- CUDA and CuDNN suitable to Caffe
- opencv

## Training
Please follow the parameter instructions in solver to set the proper learning rate. In the solver, we provide with the form as follows, `dataset task [task ] lr`. And the condition that `test_iter x test_batch_size=test_data_size` should be satisfied.

After set the right parameters, please doenload the pretrained alexnet model in ImageNet [here](http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel) to [models/bvlc_reference_caffenet](models/bvlc_reference_caffenet). The command for running code is
```
./build/tools/caffe -solver models/cdan/solver_cdan_alex.prototxt -weighths models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel -gpu gpu_id
```

## Note
- We only implement the alexnet version for caffe due to memory consumption of caffe resnet.
