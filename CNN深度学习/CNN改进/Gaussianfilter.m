% 功能：对一维信号进行高斯滤波，首尾r-1的信号不进行滤波
% r     :高斯模板的大小
% sigma :标准差
% y     :需要进行高斯滤波的序列
function y_filted = Gaussianfilter(r, sigma, y)
GaussTemp = ones(1,r*2-1);
for i=1 : r*2-1
    GaussTemp(i) = exp(-(i-r)^2/(2*sigma^2))/(sigma*sqrt(2*pi));
end

% 高斯滤波
y_filted = y;
for i = r : length(y)-r+1
    y_filted(i) = y(i-r+1 : i+r-1)*GaussTemp';
end