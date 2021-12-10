clear all;
clc;

%构建人造数据集使用的参数
true_w = [2 , -3.4];
true_b = 4.2;
%样本数量为1000
[features , labels] = synthetic_data(true_w, true_b, 1000);
%每个批量大小为10
batch_size = 10;
%初始化w和b
w = 0.01 * randn(2 , 1) + 0;
w = transpose(w);
b = 0;
%学习率0.03
lr = 0.03;
%迭代3次
num_epochs = 3;
%计数值
cnt = 0;

%num_epochs次遍历训练数据集
for m = 1 : num_epochs
    %num批
    num = size(labels , 1) / batch_size;
    %将索引打乱数序
    indices = randperm(size(labels , 1));
    
    for i = 1 : batch_size : size(labels , 1)
        %装载第i个批量
        for j = 1 : batch_size
            X(j , :) = features(indices(i + j - 1) , :);
            y(j , :) = labels(indices(i + j - 1) , :);
        end
        cnt = cnt + 1;
        l(cnt) = squared_loss(linreg(X , w , b) , y);
        %求梯度
        dw = transpose(X * transpose(w) + b - y) * X / batch_size;
        db = sum(transpose(X * transpose(w) + b - y)) / batch_size;
        %更新
        w = w - lr * dw;
        b = b - lr * db;
    end   
    
end

%绘制损失函数变化过程
plot(l);