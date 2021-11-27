import matplotlib.pyplot as plt
import torch

from synthetic_data import synthetic_data
from data_iter import data_iter
from linreg import linreg
from squared_loss import squared_loss
from sgd import sgd

true_w = torch.tensor([2, -3.4])
true_b = 4.2
#样本数量1000
features, labels = synthetic_data(true_w, true_b, 1000)
#每个批量大小为10
batch_size = 10
#初始化w和b
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
#学习率0.03
lr = 0.03
#迭代3次
num_epochs = 3
#线性回归模型
net = linreg
#损失函数
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')

plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), color = 'blue')
plt.show()