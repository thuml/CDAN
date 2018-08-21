from torchvision import datasets
import os.path as osp
import os
from PIL import Image
import torch
import numpy as np

mnist_dataset = datasets.MNIST('../data/mnist', train=True, transform=None, 
                               target_transform=None)

outdir = '../data/usps2mnist'


mnist_train_image = osp.join(outdir, 'mnist_train_image')
os.system("mkdir -p " + mnist_train_image)
with open(osp.join(outdir, 'mnist_train.txt'), 'w') as label_file:
    for i in range(len(mnist_dataset)):
        img = Image.fromarray(mnist_dataset.train_data[i].numpy().astype(np.uint8) ).convert('L')
        img.save(osp.join(mnist_train_image, '{:d}.jpg'.format(i)))
        label_file.write('/home/large_dataset/caozhangjie/cdan/data/usps2mnist/mnist_train_image/{:d}.jpg {:d}\n'.format(i, mnist_dataset.train_labels[i]))

mnist_test_image = osp.join(outdir, 'mnist_test_image')
os.system("mkdir -p " + mnist_test_image)
mnist_test_set = torch.load("/home/large_dataset/caozhangjie/cdan/data/mnist/processed/test.pt")
with open(osp.join(outdir, 'mnist_test.txt'), 'w') as label_file:
    for i in range(mnist_test_set[0].size(0)):
        img = Image.fromarray(mnist_test_set[0][i, :, :].numpy().astype(np.uint8) ).convert('L')
        img.save(osp.join(mnist_test_image, '{:d}.jpg'.format(i)))
        label_file.write('/home/large_dataset/caozhangjie/cdan/data/usps2mnist/mnist_test_image/{:d}.jpg {:d}\n'.format(i, mnist_test_set[1][i]))

usps_train_image_path = osp.join(outdir, "usps_train_image")
usps_test_image_path = osp.join(outdir, "usps_test_image")
usps_train_image = np.load("../data/usps/train_image.npy")
usps_test_image = np.load("../data/usps/test_image.npy")
usps_train_label = np.load("../data/usps/train_label.npy")
usps_test_label = np.load("../data/usps/test_label.npy")
os.system("mkdir -p " + usps_train_image_path)
with open(osp.join(outdir, 'usps_train.txt'), 'w') as label_file:
    for i in range(usps_train_image.shape[0]):
        img = Image.fromarray( (255.0*(usps_train_image[i, 0,:,:]*0.5+0.5)).astype(np.uint8) ).convert('L')
        img.save(osp.join(usps_train_image_path, '{:d}.jpg'.format(i)))
        label_file.write('/home/large_dataset/caozhangjie/cdan/data/usps2mnist/usps_train_image/{:d}.jpg {:d}\n'.format(i, usps_train_label[i]-1))
os.system("mkdir -p " + usps_test_image_path)
with open(osp.join(outdir, 'usps_test.txt'), 'w') as label_file:
    for i in range(usps_test_image.shape[0]):
        img = Image.fromarray( (255.0*(usps_test_image[i, 0,:,:]*0.5+0.5)).astype(np.uint8) ).convert('L')
        img.save(osp.join(usps_test_image_path, '{:d}.jpg'.format(i)))
        label_file.write('/home/large_dataset/caozhangjie/cdan/data/usps2mnist/usps_test_image/{:d}.jpg {:d}\n'.format(i, usps_test_label[i]-1))
