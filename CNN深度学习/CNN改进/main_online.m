%% 清空环境：
clc
clear
close
BS1 = [0, 0];
BS2 = [12, 0];
BS3 = [0, 8];
BS4 = [12, 8];
A = [BS1; BS2; BS3; BS4]; % A表示每一个已知节点构成坐标矩阵，四个
% 场景对数损耗模型参数设置：
pd0 = -40;  % pd0接收功率在d=1m时；
n = 2.2;      % n表示路径损耗因子
tol = 100; % RSSI总共测量total次
SNR = 30; % 信噪比；
%% 设定未知位置坐标
MSx = 1:0.5:10;
MSy = 1:0.5:6;
for mm = 1:length(MSx)
    for nn = 1:length(MSy)
        MSS{mm, nn} = [MSx(mm), MSy(nn)];
    end
end
MSS_1 = reshape(MSS, mm*nn, 1);
%% 算法测试
n_est = 2.2;
pd0_est = -40;
load CNNnet.mat
load CNN-Bilstm-net.mat
for MC = 1:mm*nn

    MS = MSS_1{MC};
    MS_2(MC, :) = MS;
    r1 = A - ones(4, 1)*MS;
    r2 = (sum(r1.^2, 2)).^(1/2); % 计算估计点到基站的距离；
    
    for i = 1:4
        rssi(i, :) = getRSSI(r2(i), SNR, tol); % 原始RSSI序列
        RSSI(i, :) = mean(rssi(i, 1:20)); % 得到RSSI均值
        rssi_f(i, :) = dual_filter(rssi(i, :));
    end

    %% 定位算法---
    % 模型参数估计
    d_est = 10.^((RSSI - pd0_est) / (-10 * n_est)); % 未知节点到已知节点距离
    % 最小二乘
    P_est1(MC, :) = TOALLOP(A, d_est, 1); % 最小二乘法定位
    
    % 质心定位
    [S, p] = sort(d_est);
    P_est2(MC, :) = Triangle(A(p(1), :), A(p(2), :), A(p(3), :), S(1), S(2), S(3));

    % Dual-filter-CNN
    XTest(:, :, 1, 1) = rssi_f;
    P_est3(MC, :) = predict(cnn_net, XTest(:, :, 1, 1));

    % Dual-filter-CNN-BiLSTM
    P_est4(MC, :) = predict(model, XTest(:, :, 1, 1));
  
    % 误差计算
    err1(MC) = norm(MS - P_est1(MC, :));
    err2(MC) = norm(MS - P_est2(MC, :));
    err3(MC) = norm(MS - P_est3(MC, :));
    err4(MC) = norm(MS - P_est4(MC, :));
end

%% 误差分析
% 平均\最大\最小误差
ERR = [err1', err2', err3', err4'];
err = mean(ERR);
err_max = max(ERR);
err_min = min(ERR);
err_std = std(ERR);

err_mean2 = mean(ERR .^2);
err_rms   = sqrt(err_mean2);

% 将结果输出到命令行
%methods = {'最小二乘定位', '质心定位', 'CNN定位', 'CNN-BiLSTM定位'};
%fprintf('算法统计指标：\n');
methods = {'最小二乘定位', '质心定位', 'CNN定位', 'CNN-BiLSTM定位'};
fprintf('算法统计指标：\n');
fprintf('%-12s | %8s | %8s | %8s | %8s\n', '方法', 'Min(m)', 'Max(m)', 'RMS(m)', 'Std(m)');
fprintf('-------------------------------------------------------------\n');
for k = 1:4
    fprintf('%-12s | %8.3f | %8.3f | %8.3f | %8.3f\n', ...
        methods{k}, err_min(k), err_max(k), err_rms(k), err_std(k));
end

% 计算累计误差
err_s = sort(ERR);
err_p = (1:MC)./MC;
figure
plot(err1, 'k-', 'LineWidth', 2)
hold on
plot(err2, 'b-', 'LineWidth', 2)
plot(err3, 'r-', 'LineWidth', 2)
plot(err4, 'g-', 'LineWidth', 2)
xlabel('定位点', 'FontSize', 18)
ylabel('距离误差/m', 'FontSize', 18)
legend({'最小二乘定位', '质心定位', 'CNN定位', 'CNN-BiLSTM定位'}, 'FontSize', 18)

figure
box on
grid on
hold on

% 专门为黑白印刷设计的绘图方案
plot(err_s(:,1), err_p, 'k-s', 'LineWidth', 2.5, 'MarkerSize', 10, ...
    'MarkerIndices', 1:15:length(err_p), 'MarkerFaceColor', 'k') % 实线+实心方块
plot(err_s(:,2), err_p, 'k--d', 'LineWidth', 2.5, 'MarkerSize', 10, ...
    'MarkerIndices', 5:15:length(err_p), 'MarkerFaceColor', 'w') % 虚线+空心菱形
plot(err_s(:,3), err_p, 'k-o', 'LineWidth', 2.5, 'MarkerSize', 10, ...
    'MarkerIndices', 10:15:length(err_p), 'MarkerFaceColor', [0.5 0.5 0.5]) % 实线+灰色圆圈
plot(err_s(:,4), err_p, 'k-.^', 'LineWidth', 2.5, 'MarkerSize', 10, ...
    'MarkerIndices', 15:15:length(err_p), 'MarkerFaceColor', 'k') % 点划线+实心三角

xlabel('定位误差/m', 'FontSize', 18)
ylabel('误差累积概率分布函数（CDF）', 'FontSize', 18)
legend({'最小二乘定位', '质心定位', 'CNN定位', 'CNN-BiLSTM定位'}, ...
    'FontSize', 18, 'Location', 'best', 'EdgeColor', 'none')
ylim([0, 1])

% 增强黑白对比度
set(gca, 'GridLineStyle', ':', 'GridAlpha', 0.2)  % 改用点线网格
set(gcf, 'Position', [100 100 900 600], 'Color', 'w')  % 白底大图
%set(gca, 'FontWeight', 'bold')  % 坐标轴加粗

figure
hold on
grid on
box on
plot(MS_2(:, 1), MS_2(:, 2), 'o', 'MarkerSize', 8)
plot(P_est1(:, 1), P_est1(:, 2), '*')
plot(P_est2(:, 1), P_est2(:, 2), '+')
plot(P_est3(:, 1), P_est3(:, 2), 'p')
plot(P_est4(:, 1), P_est4(:, 2), '>')
xlabel('X方向/m', 'FontSize', 18)
ylabel('Y方向/m', 'FontSize', 18)
legend({'最小二乘定位', '质心定位', 'CNN定位', 'CNN-BiLSTM定位'}, 'FontSize', 18)

figure
b = bar(err);
grid on;
ch = get(b, 'children');
set(gca, 'XTickLabel',  {'最小二乘定位', '质心定位', 'CNN定位', 'CNN-BiLSTM定位'})
set(ch, 'FaceVertexCData', [1 0 1])
xlabel('定位方式', 'FontSize', 18)
ylabel('平均误差（m）', 'FontSize', 18);

% 新建一个图，只显示20个点的定位结果
figure
hold on
grid on
box on
% 随机选择20个点
num_highlight = 20;
highlight_indices = randi(mm*nn, num_highlight, 1);

% 绘制这20个点的定位结果
plot(MS_2(highlight_indices, 1), MS_2(highlight_indices, 2), 'ko', 'MarkerSize', 8, 'DisplayName', '待定位点')
plot(P_est1(highlight_indices, 1), P_est1(highlight_indices, 2), 'k*', 'MarkerSize', 8, 'DisplayName', '最小二乘定位')
plot(P_est2(highlight_indices, 1), P_est2(highlight_indices, 2), 'k+', 'MarkerSize', 8, 'DisplayName', '质心定位')
plot(P_est3(highlight_indices, 1), P_est3(highlight_indices, 2), 'kp', 'MarkerSize', 8, 'DisplayName', 'CNN定位')
plot(P_est4(highlight_indices, 1), P_est4(highlight_indices, 2), 'k>', 'MarkerSize', 8, 'DisplayName', 'CNN-BiLSTM定位')

xlabel('X 方向/m', 'FontSize', 18)
ylabel('Y 方向/m', 'FontSize', 18)
legend('show') % 显示图例
title('20个随机点定位结果的比较', 'FontSize', 18)
%% 保存结果
Method = ["Least Squares Positioning"; "Centroid Localization"; "CNN positioning"; "CNN-BiLSTM positioning"];
error_mean = err'; % 平均误差/m
error_max = err_max'; % 最大误差
error_min = err_min'; % 最小误差
error_std = err_std'; % 误差的方差
T = table(Method, error_mean, error_max, error_min, error_std);