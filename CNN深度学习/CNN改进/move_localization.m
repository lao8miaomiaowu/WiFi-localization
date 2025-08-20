clc; clear; close all;

% 设定四个基站位置
BS1 = [0, 0]; 
BS2 = [12, 0];
BS3 = [0, 8];
BS4 = [12, 8];
A = [BS1; BS2; BS3; BS4];

% 模型参数
pd0 = -40; 
n = 2.2;
tol = 100;
SNR = 30;

% 加载CNN-BiLSTM模型
load CNN-Bilstm-net.mat % 模型变量名：model

% 构造正方形轨迹（例如：从(2,2)出发，每边4m）
side_len = 6;
step = 0.5;
x0 = 1; y0 = 1;

traj = [...
    (x0:step:x0+side_len)',           repmat(y0, length(x0:step:x0+side_len), 1); % 右
    repmat(x0+side_len, length(y0+step:step:y0+side_len), 1), (y0+step:step:y0+side_len)'; % 上
    (x0+side_len-step:-step:x0)',     repmat(y0+side_len, length(x0+side_len-step:-step:x0), 1); % 左
    repmat(x0, length(y0+side_len-step:-step:y0+step), 1), (y0+side_len-step:-step:y0+step)']; % 下

num_points = size(traj, 1);
MS_all = traj; % ground truth

% 初始化
P_est = zeros(num_points, 2);

for idx = 1:num_points
    MS = MS_all(idx, :);
    r1 = A - ones(4,1)*MS;
    r2 = sqrt(sum(r1.^2, 2)); % 到各基站距离

    for i = 1:4
        rssi(i,:) = getRSSI(r2(i), SNR, tol); % 原始RSSI
        rssi_f(i,:) = dual_filter(rssi(i,:)); % 滤波
    end

    % CNN-BiLSTM预测
    XTest(:, :, 1, 1) = rssi_f;
    P_est(idx, :) = predict(model, XTest(:, :, 1, 1));
end

% 绘图：真实轨迹 + CNN-BiLSTM定位
figure; hold on;
grid on;
box on;
ylim([0 8])
xlim([0 12])
plot(MS_all(:,1), MS_all(:,2), 'b-o', 'LineWidth', 2, 'DisplayName', '真实轨迹');
plot(P_est(:,1), P_est(:,2), 'r-*', 'LineWidth', 1.5, 'DisplayName', 'CNN-BiLSTM定位');
plot(A(:,1), A(:,2), 'ks', 'MarkerFaceColor', 'k', 'DisplayName', '基站');
legend('show');
xlabel('X方向/m', 'FontSize', 18); ylabel('Y方向/m','FontSize', 18);
title('基于CNN-BiLSTM算法的定位轨迹','FontSize', 18);
axis equal;
