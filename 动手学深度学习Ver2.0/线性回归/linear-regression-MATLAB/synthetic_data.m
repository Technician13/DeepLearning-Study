function [X ,y] = synthetic_data(w_input ,b_input , num_examples_input)
    %XΪ��������Ϊ0������Ϊ1����̬�ֲ��ģ�num_examples_input��size(true_w , 2)�е�����
    X = 1.* randn(num_examples_input ,size(w_input , 2)) + 0;
    y = X * transpose(w_input) + b_input;
    y = y + 0.01 * randn(num_examples_input , 1);
end

