# CDAN
Code release for "Conditional Domain Adversarial Network" (NIPS 2018)

## Prerequisites
- PyTorch >= 0.4.0 (with suitable CUDA and CuDNN version)
- torchvision >= 0.2.1
- Python3
- Numpy
- argparse
- PIL

## Dataset
### Digits
Processed SVHN_dataset is [here](https://drive.google.com/open?id=1Y0wT_ElbDcnFxtu25MB74npURwwijEdT). We change the original mat into images. Other transformed images are in `data/svhn2mnist` and `data/usps2mnist`. Dataset_train.txt are lists for source and target domains and Dataset_test.txt are lists for test.

### Office-31
Office-31 dataset can be found [here](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/). 

### Office-Home
Office-Home dataset can be found [here](http://hemanthdv.org/OfficeHome-Dataset/).

### VisDA-2017
VisDA 2017 dataset can be found [here](https://github.com/VisionLearningGroup/taskcv-2017-public) in the classification track.

### Image-clef
We release the Image-clef dataset we used [here](https://drive.google.com/file/d/0B9kJH0-rJ2uRS3JILThaQXJhQlk/view).

## Training (PyTorch)
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

pythonn train_image.py --gpu_id id --net ResNet50 --dset office --test_interval 500 --s_dset_path ../../data/office/amazon_list.txt --t_dset_path ../../data/office/webcam_list.txt
```
```
Office-Home

pythonn train_image.py --gpu_id id --net ResNet50 --dset office-home --test_interval 2000 --s_dset_path ../../data/office-home/Art.txt --t_dset_path ../../data/office-home/Clipart.txt
```
```
VisDA 2017

pythonn train_image.py --gpu_id id --net ResNet50 --dset visda --test_interval 5000 --s_dset_path ../../data/visda-2017/train_list.txt --t_dset_path ../../data/visda-2017/validation_list.txt
```
```
Image-clef

pythonn train_image.py --gpu_id id --net ResNet50 --dset image-clef --test_interval 500 --s_dset_path ../../data/image-clef/b_list.txt --t_dset_path ../../data/image-clef/i_list.txt
```

If you want to run the random version of CDAN, add `--random` as a parameter.

## Training (Caffe)
Under developing.

## Training (Tensorflow)
Under developing.

## Citation
If you use this code for your research, please consider citing:
```
@inproceedings{long2018conditional,
  title={Conditional adversarial domain adaptation},
  author={Long, Mingsheng and Cao, Zhangjie and Wang, Jianmin and Jordan, Michael I},
  booktitle={Advances in Neural Information Processing Systems},
  pages={1645--1655},
  year={2018}
}
```

## Contact
If you have any problem about our code, feel free to contact
- caozj@cs.stanford.edu
- longmingsheng@gmail.com

or describe your problem in Issues.
