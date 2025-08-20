
clc
clear 
close 
%% 场景布局说明：
% 仿真场景说明：12m*8m的空间内，在顶点处设置信号发射基站；每隔0.25m-0.25m的设置一个采样点；
L1 = 12;%长方形长度；
L2 = 8;%长方形宽度；
P_0 = [0,0];%参考点；
LL1 = (1:0.25:L1-1);
LL2 = (1:0.25:L2-1);
BS1=P_0;
BS2=[P_0(1)+L1,P_0(2)];
BS3=[P_0(1),P_0(2)+L2];
BS4=[P_0(1)+L1,P_0(2)+L2];
P_m=[BS1;BS2;BS3;BS4];%基站位置

%% 仿真参数
std_var=0.3;% 模型标准偏差dB
pd0=-40;  % pd0接收功率在d=1m时；
n=2.2;      % n表示路径损耗因子
total=100;% RSSI总共测量total次
SNR = 30;%信噪比；

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
             
              rssi_f(p,:) = dual_filter(rssi(p,:));% 双重滤波；
          end

        P_weizhi = [P_weizhi;P_1{i,j}];
        RSS_input(:,:,1,C) = rssi;%1,原始RSSI序列；
        RSS_input_f(:,:,1,C) =rssi_f;%2，滤波之后的RSSI序列；
    end
end
% 保存数据
save offlineData  P_weizhi RSS_input RSS_input_f

%% CNN结构
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
% 训练参数
options = trainingOptions('adam', ...
    'MaxEpochs',80, ...  
    'GradientThreshold',1, ...
    'InitialLearnRate',0.01, ...  
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',20, ... 
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');
% cnn训练：
[cnn_net, info]  = trainNetwork(RSS_input_f,P_weizhi,layers,options);
%[cnn_net1, tr1]  = trainNetwork(RSS_input,P_weizhi,layers,options);
% cnn预测：
T_sim= predict(cnn_net,RSS_input_f);

% 误差计算（估计位置和真实位置之间的距离）：
for i = 1:length(T_sim)
    err(i) = norm(T_sim(i,:) - P_weizhi(i,:));
end

mean_err = mean(err);

% 计算相关系数：
P1 = corr(P_weizhi(:,1),T_sim(:,1),'type','Pearson');
P2 = corr(P_weizhi(:,2),T_sim(:,2),'type','Pearson');
% 保存CNN网络
save CNNnet cnn_net 

exportONNXNetwork(cnn_net, 'CNN-net.onnx', 'OpsetVersion', 12);

 %% 训练过程绘图
rmse_train = sqrt(info.TrainingRMSE);
loss_train = info.TrainingLoss;
rmse_train_smooth = movmean(rmse_train, 7);

% 左侧 Y 轴：RMSE 曲线
yyaxis left
ylim([0 2])
yticks(0:0.2:2)
plot(rmse_train_smooth, 'k-', 'LineWidth', 1)
ylabel('RMSE (m)', 'FontSize', 18)

% 右侧 Y 轴：Loss 曲线
%yyaxis right
%ylim([0 20])
%yticks(0:10:20)
%plot(loss_train, 'r-', 'LineWidth', 1)
%ylabel('损失', 'FontSize', 18)
% 图形标题和标签
xlabel('迭代', 'FontSize', 18)
title('CNN离线训练过程迭代中的RMSE和损失', 'FontSize', 18)
legend({'RMSE', '损失', 'Location', 'northeast'}, 'FontSize', 18)

figure
plot(P_weizhi(:,1),T_sim(:,1),'*')
xlabel('横坐标真实值', 'FontSize', 18)
ylabel('横坐标预测值', 'FontSize', 18)
text(2,6,['相关系数=' num2str(P1)]);

figure
plot(P_weizhi(:,2),T_sim(:,2),'kp')
xlabel('纵坐标真实值', 'FontSize', 18)
ylabel('纵坐标预测值', 'FontSize', 18)
text(2,6,['相关系数=' num2str(P2)]);

figure
plot(err)
xlabel('测试点', 'FontSize', 18)
ylabel('CNN预测误差/m', 'FontSize', 18)

figure
hold on 
box on
plot(P_weizhi(:,1),P_weizhi(:,2),'bo')
plot(P_m(:,1),P_m(:,2),'rp')
legend({'采样点','锚节点'}, 'FontSize', 18)
xlabel('长度方向/m', 'FontSize', 18)
ylabel('宽度方向/m', 'FontSize', 18)


fprintf('CNN-BiLSTM 预测误差平均值：%.3f m\n', mean_err);