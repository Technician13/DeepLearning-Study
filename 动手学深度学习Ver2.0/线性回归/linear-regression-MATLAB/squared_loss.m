function [loss] = squared_loss(y_hat_input , y_input)
    %¾ù·½ËğÊ§
    loss = (y_hat_input - y_input).^2 / 2;
    loss = sum(loss);
end

