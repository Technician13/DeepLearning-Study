function [X ,y] = synthetic_data(w_input ,b_input , num_examples_input)
    %X为服从期望为0，方差为1的正态分布的，num_examples_input，size(true_w , 2)列的张量
    X = 1.* randn(num_examples_input ,size(w_input , 2)) + 0;
    y = X * transpose(w_input) + b_input;
    y = y + 0.01 * randn(num_examples_input , 1);
end

