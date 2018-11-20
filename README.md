# CDAN
Code release for "Conditional Domain Adversarial Network" (NIPS 2018)

## Dataset
Processed SVHN_dataset is [here](https://drive.google.com/open?id=1Y0wT_ElbDcnFxtu25MB74npURwwijEdT). If you hope to use this dataset, please cite the original paper of the dataset.
```
@inproceedings{cite:NIPS11SVHN,
  title={Reading digits in natural images with unsupervised feature learning},
  author={Netzer, Yuval and Wang, Tao and Coates, Adam and Bissacco, Alessandro and Wu, Bo and Ng, Andrew Y},
  booktitle={NIPS workshop on deep learning and unsupervised feature learning},
  volume={2011},
  pages={5},
  year={2011}
}
```

## Train
```
pythonn train.py --gpu_id id --net ResNet50 --dset office --test_interval 500 --s_dset_path ../data/office/amazon_list.txt --t_dset_path ../data/office/webcam_list.txt
pythonn train.py --gpu_id id --net ResNet50 --dset office-home --test_interval 2000 --s_dset_path ../data/office-home/Art.txt --t_dset_path ../data/office-home/Clipart.txt
```
