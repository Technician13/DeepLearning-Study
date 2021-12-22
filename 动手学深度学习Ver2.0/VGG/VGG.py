import torch
from torch import nn
from d2l import torch as d2l

from vgg_block import vgg_block
from vgg_net import vgg

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

net = vgg(conv_arch)

ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)

lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())