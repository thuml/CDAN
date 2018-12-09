# CDAN implemneted in PyTorch

## Prerequisites
- PyTorch >= 0.4.0 (with suitable CUDA and CuDNN version)
- torchvision >= 0.2.1
- Python3
- Numpy
- argparse
- PIL

## Training
All the parameters are set to optimal in our experiments. The following are the command for each task. The test_interval can be changed, which is the number of iterations between near test.
```
SVHN->MNIST
python train_svhnmnist.py --gpu_id id --epochs 50

USPS->MNIST
python train_uspsmnist.py --gpu_id id --epochs 50 --task USPS2MNIST

MNIST->USPS
python train_uspsmnist.py --gpu_id id --epochs 50 --task MNIST2USPS
```
```
Office-31

pythonn train_image.py --gpu_id id --net ResNet50 --dset office --test_interval 500 --s_dset_path ../data/office/amazon_list.txt --t_dset_path ../data/office/webcam_list.txt
```
```
Office-Home

pythonn train_image.py --gpu_id id --net ResNet50 --dset office-home --test_interval 2000 --s_dset_path ../data/office-home/Art.txt --t_dset_path ../data/office-home/Clipart.txt
```
```
VisDA 2017

pythonn train_image.py --gpu_id id --net ResNet50 --dset visda --test_interval 5000 --s_dset_path ../data/visda-2017/train_list.txt --t_dset_path ../data/visda-2017/validation_list.txt
```
```
Image-clef

pythonn train_image.py --gpu_id id --net ResNet50 --dset image-clef --test_interval 500 --s_dset_path ../data/image-clef/b_list.txt --t_dset_path ../data/image-clef/i_list.txt
```

If you want to run the random version of CDAN, add `--random` as a parameter.

## Note
- The alexnet version for PyTorch is under development. We plan to use alexnet [here](https://github.com/jiecaoyu/pytorch_imagenet)
