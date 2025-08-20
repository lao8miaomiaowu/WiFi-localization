
clc
clear 
close 
%% ��������˵����
% ���泡��˵����12m*8m�Ŀռ��ڣ��ڶ��㴦�����źŷ����վ��ÿ��0.25m-0.25m������һ�������㣻
L1 = 12;%�����γ��ȣ�
L2 = 8;%�����ο�ȣ�
P_0 = [0,0];%�ο��㣻
LL1 = (1:0.25:L1-1);
LL2 = (1:0.25:L2-1);
BS1=P_0;
BS2=[P_0(1)+L1,P_0(2)];
BS3=[P_0(1),P_0(2)+L2];
BS4=[P_0(1)+L1,P_0(2)+L2];
P_m=[BS1;BS2;BS3;BS4];%��վλ��

%% �������
std_var=0.3;% ģ�ͱ�׼ƫ��dB
pd0=-40;  % pd0���չ�����d=1mʱ��
n=2.2;      % n��ʾ·���������
total=100;% RSSI�ܹ�����total��
SNR = 30;%����ȣ�

%% simu
P_weizhi=[];
C = 0;
for i = 1:length(LL1)
    for j = 1:length(LL2)
        C = C+1;
        P_1{i,j} = P_0+[LL1(i),LL2(j)];
        r1=P_m-P_1{i,j};
        r2=(sum(r1.^2,2)).^(1/2);
        d_true{i,j} = r2;

          for p=1:4
              rssi(p,:)=getRSSI(r2(p),SNR,total);
             
              rssi_f(p,:) = dual_filter(rssi(p,:));% ˫���˲���
          end

        P_weizhi = [P_weizhi;P_1{i,j}];
        RSS_input(:,:,1,C) = rssi;%1,ԭʼRSSI���У�
        RSS_input_f(:,:,1,C) =rssi_f;%2���˲�֮���RSSI���У�
    end
end
% ��������
save offlineData  P_weizhi RSS_input RSS_input_f

%% CNN�ṹ
rng(0)

layers = [
    imageInputLayer([p total 1])
    
    convolution2dLayer(3,8,Padding="same")
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,Stride=2)
    
    convolution2dLayer(3,16,Padding="same")
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,Stride=2)
    
    convolution2dLayer(3,32,Padding="same")
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(2)
    regressionLayer];
% ѵ������
options = trainingOptions('adam', ...
    'MaxEpochs',80, ...  
    'GradientThreshold',1, ...
    'InitialLearnRate',0.01, ...  
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',20, ... 
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');
% cnnѵ����
[cnn_net, info]  = trainNetwork(RSS_input_f,P_weizhi,layers,options);
%[cnn_net1, tr1]  = trainNetwork(RSS_input,P_weizhi,layers,options);
% cnnԤ�⣺
T_sim= predict(cnn_net,RSS_input_f);

% �����㣨����λ�ú���ʵλ��֮��ľ��룩��
for i = 1:length(T_sim)
    err(i) = norm(T_sim(i,:) - P_weizhi(i,:));
end

mean_err = mean(err);

% �������ϵ����
P1 = corr(P_weizhi(:,1),T_sim(:,1),'type','Pearson');
P2 = corr(P_weizhi(:,2),T_sim(:,2),'type','Pearson');
% ����CNN����
save CNNnet cnn_net 

exportONNXNetwork(cnn_net, 'CNN-net.onnx', 'OpsetVersion', 12);

 %% ѵ�����̻�ͼ
rmse_train = sqrt(info.TrainingRMSE);
loss_train = info.TrainingLoss;
rmse_train_smooth = movmean(rmse_train, 7);

% ��� Y �᣺RMSE ����
yyaxis left
ylim([0 2])
yticks(0:0.2:2)
plot(rmse_train_smooth, 'k-', 'LineWidth', 1)
ylabel('RMSE (m)', 'FontSize', 18)

% �Ҳ� Y �᣺Loss ����
%yyaxis right
%ylim([0 20])
%yticks(0:10:20)
%plot(loss_train, 'r-', 'LineWidth', 1)
%ylabel('��ʧ', 'FontSize', 18)
% ͼ�α���ͱ�ǩ
xlabel('����', 'FontSize', 18)
title('CNN����ѵ�����̵����е�RMSE����ʧ', 'FontSize', 18)
legend({'RMSE', '��ʧ', 'Location', 'northeast'}, 'FontSize', 18)

figure
plot(P_weizhi(:,1),T_sim(:,1),'*')
xlabel('��������ʵֵ', 'FontSize', 18)
ylabel('������Ԥ��ֵ', 'FontSize', 18)
text(2,6,['���ϵ��=' num2str(P1)]);

figure
plot(P_weizhi(:,2),T_sim(:,2),'kp')
xlabel('��������ʵֵ', 'FontSize', 18)
ylabel('������Ԥ��ֵ', 'FontSize', 18)
text(2,6,['���ϵ��=' num2str(P2)]);

figure
plot(err)
xlabel('���Ե�', 'FontSize', 18)
ylabel('CNNԤ�����/m', 'FontSize', 18)

figure
hold on 
box on
plot(P_weizhi(:,1),P_weizhi(:,2),'bo')
plot(P_m(:,1),P_m(:,2),'rp')
legend({'������','ê�ڵ�'}, 'FontSize', 18)
xlabel('���ȷ���/m', 'FontSize', 18)
ylabel('��ȷ���/m', 'FontSize', 18)


fprintf('CNN-BiLSTM Ԥ�����ƽ��ֵ��%.3f m\n', mean_err);