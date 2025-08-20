clc
clear
close all;
load offlineData.mat
%% cnn-bilstm-attention
inputSize = 4;
numResponses=2;

%%  建立模型
layers = [
    imageInputLayer([4 100 1])
    
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
    flattenLayer
    bilstmLayer(256,"OutputMode","last")
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

% CNN-Bilstm-attention训练
for i =1:289
    
    input_seq{i,1} = reshape(RSS_input_f(:,1,1,i),4,1);
end
[model, info] = trainNetwork(RSS_input_f, P_weizhi, layers, options);

% CNN-BiLSTM预测
T_sim = predict(model,RSS_input_f);

% 误差计算（估计位置和真实位置之间的距离）
for i = 1:length(T_sim)
    err(i) = norm(T_sim(i,:) - P_weizhi(i,:));
end

mean_err = mean(err);

% 计算相关系数
P1 = corr(P_weizhi(:,1), T_sim(:,1), 'type', 'Pearson');
P2 = corr(P_weizhi(:,2), T_sim(:,2), 'type', 'Pearson');

% 保存CNN-Bilstm-attention网络
save CNN-Bilstm-net model

exportONNXNetwork(model, 'CNN-Bilstm-net.onnx', 'OpsetVersion', 12);

 %% 训练过程绘图
rmse_train = sqrt(info.TrainingRMSE);
loss_train = info.TrainingLoss;
rmse_train_smooth = movmean(rmse_train, 7);
% 左侧 Y 轴：RMSE 曲线
yyaxis left
ylim([0 2])
yticks(0:0.2:2)
plot(rmse_train_smooth, 'b-', 'LineWidth', 1)
ylabel('RMSE (m)', 'FontSize', 18)
% 右侧 Y 轴：Loss 曲线
%yyaxis right
%ylim([0 20])
%yticks(0:10:20)
%plot(loss_train, 'r-', 'LineWidth', 1)
%ylabel('损失', 'FontSize', 18)
% 图形标题和标签
xlabel('迭代', 'FontSize', 18)
title('CNN-BiLSTM离线训练过程迭代中的RMSE和损失', 'FontSize', 18)
legend({'RMSE', '损失', 'Location', 'northeast'}, 'FontSize', 18)

figure
plot(P_weizhi(:,1), T_sim(:,1), '*')
xlabel('横坐标真实值', 'FontSize', 18)
ylabel('横坐标预测值', 'FontSize', 18)
text(2,6,['相关系数=' num2str(P1)]);

figure
plot(P_weizhi(:,2), T_sim(:,2), 'kp')
xlabel('纵坐标真实值', 'FontSize',18)
ylabel('纵坐标预测值', 'FontSize',18)
text(2,6,['相关系数=' num2str(P2)] );

figure
plot(err)
xlabel('测试点', 'FontSize',18)
ylabel('CNN-BiLSTM预测误差/m', 'FontSize',18)

fprintf('CNN-BiLSTM 预测误差平均值：%.3f m\n', mean_err);