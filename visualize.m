clear; clc; close all;
gtData = readmatrix('salinity_slice.txt')';
gpData = readmatrix('salinity_slice_after.txt');

xb = 0:10:100;
yb = 0:10:100;

[XB, YB] = meshgrid(xb, yb);

x = 1:0.1:99;
y = x;
for i = 1:length(x)
    y(i) = 50 + 49.5 * sin(4 * pi * (x(i) - 1) / (99 - 1));
end

z = interp2(xb, yb, gpData, x, y) + 0.01;

figure;
surf(XB, YB, gtData(1:11, 1:11));
title('Ground-Truth');

figure;
hold on;
grid on;
surf(XB, YB, gpData);
plot3(x, y, z, 'k', 'LineWidth', 5);
hold off;
title('Gaussian Process');

z = interp2(xb, yb, gtData(1:11, 1:11) - gpData, x, y) + 0.01;

figure;
hold on;
grid on;
surf(XB, YB, gtData(1:11, 1:11) - gpData);
plot3(x, y, z, 'k', 'LineWidth', 5);
hold off;
title('Delta GT and GP');