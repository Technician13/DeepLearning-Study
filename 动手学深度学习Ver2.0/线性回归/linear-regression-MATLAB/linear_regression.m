clear all;
clc;

%�����������ݼ�ʹ�õĲ���
true_w = [2 , -3.4];
true_b = 4.2;
%��������Ϊ1000
[features , labels] = synthetic_data(true_w, true_b, 1000);
%ÿ��������СΪ10
batch_size = 10;
%��ʼ��w��b
w = 0.01 * randn(2 , 1) + 0;
w = transpose(w);
b = 0;
%ѧϰ��0.03
lr = 0.03;
%����3��
num_epochs = 3;
%����ֵ
cnt = 0;

%num_epochs�α���ѵ�����ݼ�
for m = 1 : num_epochs
    %num��
    num = size(labels , 1) / batch_size;
    %��������������
    indices = randperm(size(labels , 1));
    
    for i = 1 : batch_size : size(labels , 1)
        %װ�ص�i������
        for j = 1 : batch_size
            X(j , :) = features(indices(i + j - 1) , :);
            y(j , :) = labels(indices(i + j - 1) , :);
        end
        cnt = cnt + 1;
        l(cnt) = squared_loss(linreg(X , w , b) , y);
        %���ݶ�
        dw = transpose(X * transpose(w) + b - y) * X / batch_size;
        db = sum(transpose(X * transpose(w) + b - y)) / batch_size;
        %����
        w = w - lr * dw;
        b = b - lr * db;
    end   
    
end

%������ʧ�����仯����
plot(l);