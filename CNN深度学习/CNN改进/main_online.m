%% ��ջ�����
clc
clear
close
BS1 = [0, 0];
BS2 = [12, 0];
BS3 = [0, 8];
BS4 = [12, 8];
A = [BS1; BS2; BS3; BS4]; % A��ʾÿһ����֪�ڵ㹹����������ĸ�
% �����������ģ�Ͳ������ã�
pd0 = -40;  % pd0���չ�����d=1mʱ��
n = 2.2;      % n��ʾ·���������
tol = 100; % RSSI�ܹ�����total��
SNR = 30; % ����ȣ�
%% �趨δ֪λ������
MSx = 1:0.5:10;
MSy = 1:0.5:6;
for mm = 1:length(MSx)
    for nn = 1:length(MSy)
        MSS{mm, nn} = [MSx(mm), MSy(nn)];
    end
end
MSS_1 = reshape(MSS, mm*nn, 1);
%% �㷨����
n_est = 2.2;
pd0_est = -40;
load CNNnet.mat
load CNN-Bilstm-net.mat
for MC = 1:mm*nn

    MS = MSS_1{MC};
    MS_2(MC, :) = MS;
    r1 = A - ones(4, 1)*MS;
    r2 = (sum(r1.^2, 2)).^(1/2); % ������Ƶ㵽��վ�ľ��룻
    
    for i = 1:4
        rssi(i, :) = getRSSI(r2(i), SNR, tol); % ԭʼRSSI����
        RSSI(i, :) = mean(rssi(i, 1:20)); % �õ�RSSI��ֵ
        rssi_f(i, :) = dual_filter(rssi(i, :));
    end

    %% ��λ�㷨---
    % ģ�Ͳ�������
    d_est = 10.^((RSSI - pd0_est) / (-10 * n_est)); % δ֪�ڵ㵽��֪�ڵ����
    % ��С����
    P_est1(MC, :) = TOALLOP(A, d_est, 1); % ��С���˷���λ
    
    % ���Ķ�λ
    [S, p] = sort(d_est);
    P_est2(MC, :) = Triangle(A(p(1), :), A(p(2), :), A(p(3), :), S(1), S(2), S(3));

    % Dual-filter-CNN
    XTest(:, :, 1, 1) = rssi_f;
    P_est3(MC, :) = predict(cnn_net, XTest(:, :, 1, 1));

    % Dual-filter-CNN-BiLSTM
    P_est4(MC, :) = predict(model, XTest(:, :, 1, 1));
  
    % ������
    err1(MC) = norm(MS - P_est1(MC, :));
    err2(MC) = norm(MS - P_est2(MC, :));
    err3(MC) = norm(MS - P_est3(MC, :));
    err4(MC) = norm(MS - P_est4(MC, :));
end

%% ������
% ƽ��\���\��С���
ERR = [err1', err2', err3', err4'];
err = mean(ERR);
err_max = max(ERR);
err_min = min(ERR);
err_std = std(ERR);

err_mean2 = mean(ERR .^2);
err_rms   = sqrt(err_mean2);

% ����������������
%methods = {'��С���˶�λ', '���Ķ�λ', 'CNN��λ', 'CNN-BiLSTM��λ'};
%fprintf('�㷨ͳ��ָ�꣺\n');
methods = {'��С���˶�λ', '���Ķ�λ', 'CNN��λ', 'CNN-BiLSTM��λ'};
fprintf('�㷨ͳ��ָ�꣺\n');
fprintf('%-12s | %8s | %8s | %8s | %8s\n', '����', 'Min(m)', 'Max(m)', 'RMS(m)', 'Std(m)');
fprintf('-------------------------------------------------------------\n');
for k = 1:4
    fprintf('%-12s | %8.3f | %8.3f | %8.3f | %8.3f\n', ...
        methods{k}, err_min(k), err_max(k), err_rms(k), err_std(k));
end

% �����ۼ����
err_s = sort(ERR);
err_p = (1:MC)./MC;
figure
plot(err1, 'k-', 'LineWidth', 2)
hold on
plot(err2, 'b-', 'LineWidth', 2)
plot(err3, 'r-', 'LineWidth', 2)
plot(err4, 'g-', 'LineWidth', 2)
xlabel('��λ��', 'FontSize', 18)
ylabel('�������/m', 'FontSize', 18)
legend({'��С���˶�λ', '���Ķ�λ', 'CNN��λ', 'CNN-BiLSTM��λ'}, 'FontSize', 18)

figure
box on
grid on
hold on

% ר��Ϊ�ڰ�ӡˢ��ƵĻ�ͼ����
plot(err_s(:,1), err_p, 'k-s', 'LineWidth', 2.5, 'MarkerSize', 10, ...
    'MarkerIndices', 1:15:length(err_p), 'MarkerFaceColor', 'k') % ʵ��+ʵ�ķ���
plot(err_s(:,2), err_p, 'k--d', 'LineWidth', 2.5, 'MarkerSize', 10, ...
    'MarkerIndices', 5:15:length(err_p), 'MarkerFaceColor', 'w') % ����+��������
plot(err_s(:,3), err_p, 'k-o', 'LineWidth', 2.5, 'MarkerSize', 10, ...
    'MarkerIndices', 10:15:length(err_p), 'MarkerFaceColor', [0.5 0.5 0.5]) % ʵ��+��ɫԲȦ
plot(err_s(:,4), err_p, 'k-.^', 'LineWidth', 2.5, 'MarkerSize', 10, ...
    'MarkerIndices', 15:15:length(err_p), 'MarkerFaceColor', 'k') % �㻮��+ʵ������

xlabel('��λ���/m', 'FontSize', 18)
ylabel('����ۻ����ʷֲ�������CDF��', 'FontSize', 18)
legend({'��С���˶�λ', '���Ķ�λ', 'CNN��λ', 'CNN-BiLSTM��λ'}, ...
    'FontSize', 18, 'Location', 'best', 'EdgeColor', 'none')
ylim([0, 1])

% ��ǿ�ڰ׶Աȶ�
set(gca, 'GridLineStyle', ':', 'GridAlpha', 0.2)  % ���õ�������
set(gcf, 'Position', [100 100 900 600], 'Color', 'w')  % �׵״�ͼ
%set(gca, 'FontWeight', 'bold')  % ������Ӵ�

figure
hold on
grid on
box on
plot(MS_2(:, 1), MS_2(:, 2), 'o', 'MarkerSize', 8)
plot(P_est1(:, 1), P_est1(:, 2), '*')
plot(P_est2(:, 1), P_est2(:, 2), '+')
plot(P_est3(:, 1), P_est3(:, 2), 'p')
plot(P_est4(:, 1), P_est4(:, 2), '>')
xlabel('X����/m', 'FontSize', 18)
ylabel('Y����/m', 'FontSize', 18)
legend({'��С���˶�λ', '���Ķ�λ', 'CNN��λ', 'CNN-BiLSTM��λ'}, 'FontSize', 18)

figure
b = bar(err);
grid on;
ch = get(b, 'children');
set(gca, 'XTickLabel',  {'��С���˶�λ', '���Ķ�λ', 'CNN��λ', 'CNN-BiLSTM��λ'})
set(ch, 'FaceVertexCData', [1 0 1])
xlabel('��λ��ʽ', 'FontSize', 18)
ylabel('ƽ����m��', 'FontSize', 18);

% �½�һ��ͼ��ֻ��ʾ20����Ķ�λ���
figure
hold on
grid on
box on
% ���ѡ��20����
num_highlight = 20;
highlight_indices = randi(mm*nn, num_highlight, 1);

% ������20����Ķ�λ���
plot(MS_2(highlight_indices, 1), MS_2(highlight_indices, 2), 'ko', 'MarkerSize', 8, 'DisplayName', '����λ��')
plot(P_est1(highlight_indices, 1), P_est1(highlight_indices, 2), 'k*', 'MarkerSize', 8, 'DisplayName', '��С���˶�λ')
plot(P_est2(highlight_indices, 1), P_est2(highlight_indices, 2), 'k+', 'MarkerSize', 8, 'DisplayName', '���Ķ�λ')
plot(P_est3(highlight_indices, 1), P_est3(highlight_indices, 2), 'kp', 'MarkerSize', 8, 'DisplayName', 'CNN��λ')
plot(P_est4(highlight_indices, 1), P_est4(highlight_indices, 2), 'k>', 'MarkerSize', 8, 'DisplayName', 'CNN-BiLSTM��λ')

xlabel('X ����/m', 'FontSize', 18)
ylabel('Y ����/m', 'FontSize', 18)
legend('show') % ��ʾͼ��
title('20������㶨λ����ıȽ�', 'FontSize', 18)
%% ������
Method = ["Least Squares Positioning"; "Centroid Localization"; "CNN positioning"; "CNN-BiLSTM positioning"];
error_mean = err'; % ƽ�����/m
error_max = err_max'; % ������
error_min = err_min'; % ��С���
error_std = err_std'; % ���ķ���
T = table(Method, error_mean, error_max, error_min, error_std);