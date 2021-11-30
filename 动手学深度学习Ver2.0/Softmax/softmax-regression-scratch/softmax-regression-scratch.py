from d2l import torch as d2l

from softmax import net
from softmax import cross_entropy
from softmax import W
from softmax import b

from train import train_ch3
from train import updater

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)