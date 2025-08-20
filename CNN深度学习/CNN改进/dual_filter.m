function y = dual_filter(x)
mu = 5;
sig = 1;
y_gs = Gaussianfilter(mu,sig, x);% 高斯滤波
% KF参数设置：
delta_w=1e-3; %如果增大这个参数，目标真实轨迹就是曲线了
Q=delta_w*diag(1) ; % 过程噪声方差
R=0.5*eye(1);  %观测噪声方差
F=1; %状态转移矩阵 
H=1; % 观测矩阵
P0=1;% 协方差矩阵
tol = length(x);
Xkf=zeros(1,tol);
Xkf(:,1)=y_gs(1); %Kalman滤波状态初始化

for j = 2:tol
    Xn=F*Xkf(:,j-1);   %预测
    P1=F*P0*F'+Q;    %预测误差协方差
    K=P1*H'*inv(H*P1*H'+R);   %增益
    Xkf(:,j)=Xn+K*(y_gs(:,j)-H*Xn); %状态更新
    P0=(eye(1)-K*H)*P1;  %滤波误差协方差更新
end
y = Xkf;
end
