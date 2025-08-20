
%% ��ջ�����
clc
clear all
close all
BS1=[0,0];
BS2=[12,0];
BS3=[12,8];
BS4=[0,8];
A=[BS1;BS2;BS3;BS4]; % A��ʾÿһ����֪�ڵ㹹����������ĸ�����λ��Ϊ�ű�λ�ã�
% �����������ģ�Ͳ������ã�
pd0=-40;  % pd0���չ�����d=1mʱ��
n=2.2;      % n��ʾ·���������
tol=100;% RSSI�ܹ�����total��
SNR = 30;%����ȣ�
% KF�������ã�
delta_w=1e-2; %����������������Ŀ����ʵ�켣����������
Q=delta_w*diag(1) ; % ������������
R=0.5*eye(1);  %�۲���������
F=1; %״̬ת�ƾ��� 
H=1; % �۲����
P0=1;% Э�������
Xkf=zeros(4,tol);
Xkf(:,1)=1; %Kalman�˲�״̬��ʼ��

%% �趨δ֪λ������

MS=[5,5];%�趨δ֪λ������

%% �㷨����

    r1=A-ones(4,1)*MS;
    r2=(sum(r1.^2,2)).^(1/2);% ������Ƶ㵽��վ�ľ��룻
    
        for i=1:4 % i�ռ����ĵ�i���ű괦��RSSI�ź�
              rssi(i,:)=getRSSI(r2(i),SNR,tol);% ԭʼRSSI����
              rssi_0(i,:)=rssi(i,:);
             % ��˹�˲���
             y_gs(i,:) = Gaussianfilter(5,1, rssi_0(i,:));% ��˹�˲�
             % �������˲�
              Xkf(:,1)=rssi(i,1);
              for j = 2:tol
                    Xn=F*Xkf(i,j-1);   %Ԥ��
                    P1=F*P0*F'+Q;    %Ԥ�����Э����
                    K=P1*H'*inv(H*P1*H'+R);   %����
                    Xkf(i,j)=Xn+K*(rssi(i,j)-H*Xn); %״̬����
                    P0=(eye(1)-K*H)*P1;  %�˲����Э�������
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
               legend('ԭʼRSSI����','��˹�˲�','�������˲�','˫���˲�') 
               title(['��',num2str(i),'·�ű��RSSI�ź�'])
               ylabel('dBm')
               xlabel('RSSI����')
               hold off
        end