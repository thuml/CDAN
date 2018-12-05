from torchvision import datasets
import os.path as osp
import os
from PIL import Image
import torch

mnist_dataset = datasets.MNIST('../data/mnist', train=True, transform=None, 
                               target_transform=None)
svhn_dataset = datasets.SVHN('../data/svhn', split='train', transform=None, 
                             target_transform=None)
outdir = '../data/svhn2mnist'

mnist_train_image = osp.join(outdir, 'mnist_train_image')
os.system("mkdir -p " + mnist_train_image)
with open(osp.join(outdir, 'mnist_train.txt'), 'w') as label_file:
    for i in range(len(mnist_dataset)):
        img = Image.fromarray(mnist_dataset.train_data[i].numpy())
        img = img.resize([32,32])
        img = img.convert('RGB')
        img.save(osp.join(mnist_train_image, '{:d}.png'.format(i)))
        label_file.write('{:d}.png {:d}\n'.format(i, mnist_dataset.train_labels[i]))

mnist_test_image = osp.join(outdir, 'mnist_test_image')
os.system("mkdir -p " + mnist_test_image)
mnist_test_set = torch.load("/home/large_dataset/caozhangjie/cdan/data/mnist/processed/test.pt")
with open(osp.join(outdir, 'mnist_test.txt'), 'w') as label_file:
    for i in range(mnist_test_set[0].size(0)):
        img = Image.fromarray(mnist_test_set[0][i, :, :].numpy())
        img = img.resize([32,32])
        img = img.convert('RGB')
        img.save(osp.join(mnist_test_image, '{:d}.png'.format(i)))
        label_file.write('{:d}.png {:d}\n'.format(i, mnist_test_set[1][i]))
     
svhn_image = osp.join(outdir, 'svhn_image')
os.system("mkdir -p " + svhn_image)
svhn_labels = svhn_dataset.labels.flatten()
with open(osp.join(outdir, 'svhn.txt'), 'w') as label_file:
    for i in range(len(svhn_dataset)):
        img = Image.fromarray(svhn_dataset.data[i].transpose(1,2,0))
        img.save(osp.join(svhn_image, '{:d}.png'.format(i)))
        label_file.write('{:d}.png {:d}\n'.format(i, svhn_labels[i]))
