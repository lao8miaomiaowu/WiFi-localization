
%% 清空环境：
clc
clear all
close all
BS1=[0,0];
BS2=[12,0];
BS3=[12,8];
BS4=[0,8];
A=[BS1;BS2;BS3;BS4]; % A表示每一个已知节点构成坐标矩阵，四个顶点位置为信标位置；
% 场景对数损耗模型参数设置：
pd0=-40;  % pd0接收功率在d=1m时；
n=2.2;      % n表示路径损耗因子
tol=100;% RSSI总共测量total次
SNR = 30;%信噪比；
% KF参数设置：
delta_w=1e-2; %如果增大这个参数，目标真实轨迹就是曲线了
Q=delta_w*diag(1) ; % 过程噪声方差
R=0.5*eye(1);  %观测噪声方差
F=1; %状态转移矩阵 
H=1; % 观测矩阵
P0=1;% 协方差矩阵
Xkf=zeros(4,tol);
Xkf(:,1)=1; %Kalman滤波状态初始化

%% 设定未知位置坐标

MS=[5,5];%设定未知位置坐标

%% 算法测试

    r1=A-ones(4,1)*MS;
    r2=(sum(r1.^2,2)).^(1/2);% 计算估计点到基站的距离；
    
        for i=1:4 % i收集到的第i个信标处的RSSI信号
              rssi(i,:)=getRSSI(r2(i),SNR,tol);% 原始RSSI序列
              rssi_0(i,:)=rssi(i,:);
             % 高斯滤波：
             y_gs(i,:) = Gaussianfilter(5,1, rssi_0(i,:));% 高斯滤波
             % 卡尔曼滤波
              Xkf(:,1)=rssi(i,1);
              for j = 2:tol
                    Xn=F*Xkf(i,j-1);   %预测
                    P1=F*P0*F'+Q;    %预测误差协方差
                    K=P1*H'*inv(H*P1*H'+R);   %增益
                    Xkf(i,j)=Xn+K*(rssi(i,j)-H*Xn); %状态更新
                    P0=(eye(1)-K*H)*P1;  %滤波误差协方差更新
              end
            y_dl(i,:) = dual_filter(rssi_0(i,:));
             
               figure
               hold on
               box on
               grid on
               plot(1:tol,rssi_0(i,:),'LineWidth',1.1,'LineStyle','-','Color','b')
               plot(1:tol,y_gs(i,:),'LineWidth',1.1,'LineStyle','--','Color','k')
               plot(1:tol,Xkf(i,:),'LineWidth',1.1,'LineStyle','-.','Color','r')
               plot(1:tol,y_dl(i,:),'LineWidth',1.1,'LineStyle',':','Color','g')
               legend('原始RSSI序列','高斯滤波','卡尔曼滤波','双重滤波') 
               title(['第',num2str(i),'路信标的RSSI信号'])
               ylabel('dBm')
               xlabel('RSSI序列')
               hold off
        end