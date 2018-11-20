# CDAN
Code release for "Conditional Domain Adversarial Network" (NIPS 2017)

## Dataset
Processed SVHN_dataset is [here]().

## Train
```
pythonn train.py --gpu_id id --net ResNet50 --dset office --test_interval 500 --s_dset_path ../data/office/amazon_list.txt --t_dset_path ../data/office/webcam_list.txt
pythonn train.py --gpu_id id --net ResNet50 --dset office-home --test_interval 2000 --s_dset_path ../data/office-home/Art.txt --t_dset_path ../data/office-home/Clipart.txt
```
