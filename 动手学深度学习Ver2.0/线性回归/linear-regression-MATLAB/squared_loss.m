function [loss] = squared_loss(y_hat_input , y_input)
    %������ʧ
    loss = (y_hat_input - y_input).^2 / 2;
    loss = sum(loss);
end

