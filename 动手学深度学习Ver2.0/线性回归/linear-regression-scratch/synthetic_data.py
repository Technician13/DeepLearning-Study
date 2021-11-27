import torch

def synthetic_data(w, b, num_examples):  
    """生成 y = Xw + b + 噪声。"""
    #X为服从期望为0，方差为1的正态分布的，num_examples行，len(w)列的张量
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    #y张量添加服从期望为0，方差为0.01的正态分布的噪声
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))
