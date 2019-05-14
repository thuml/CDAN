# CDAN
Code release for ["Conditional Adversarial Domain Adaptation"](https://papers.nips.cc/paper/7436-conditional-adversarial-domain-adaptation) (NIPS 2018)

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

## Training
Training instructions for Caffe and PyTorch are in the `README.md` in [caffe](caffe) and [pytorch](pytorch) respectively.

Tensorflow version is under developing.

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
- youkaichao@gmail.com
- shuyang5656@gmail.com
- longmingsheng@gmail.com

or describe your problem in Issues.
