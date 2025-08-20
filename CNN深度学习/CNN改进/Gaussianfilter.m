% ���ܣ���һά�źŽ��и�˹�˲�����βr-1���źŲ������˲�
% r     :��˹ģ��Ĵ�С
% sigma :��׼��
% y     :��Ҫ���и�˹�˲�������
function y_filted = Gaussianfilter(r, sigma, y)
GaussTemp = ones(1,r*2-1);
for i=1 : r*2-1
    GaussTemp(i) = exp(-(i-r)^2/(2*sigma^2))/(sigma*sqrt(2*pi));
end

% ��˹�˲�
y_filted = y;
for i = r : length(y)-r+1
    y_filted(i) = y(i-r+1 : i+r-1)*GaussTemp';
end