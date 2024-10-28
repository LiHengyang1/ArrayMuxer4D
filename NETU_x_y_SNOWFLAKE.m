clear;
close all;
lambda = 1064e-6;
f = 400;
z = f;
N = 1000;
pixel = 0.0125;
L0 = N * pixel;
k = 2 * pi / lambda;
x = linspace(-L0 / 2 + pixel / 2,L0 / 2 - pixel / 2,N);
[x,y] = meshgrid(x,x);
[theta,r] = cart2pol(x,y);
theta = theta + pi;
lensphase = exp(-1i * k * (x.^2 + y.^2) / 2 / f);
r0 = 6.25;

furiournumber_A = 32;
furiournumber_B = 16;
furiournumber_C = 16;
furiournumber_D = 16;

delta_A = 30 * lambda * f / L0 / 1;
delta_B = delta_A * 1.2;
delta_C = 3;
delta_D = 9;
differencephase_A = k * x / z * delta_A;
differencephase_B = k * y / z * delta_B;
differencephase_C = 6 * (sign(sin(8 * r)-0.1)) + 1.5*r;
differencephase_D = sin(6 * theta) * 0.8 .* r;

differencephase_A_mapped = floor(mod(differencephase_A,2 * pi) / 2 / pi * furiournumber_A) + 1;
differencephase_B_mapped = floor(mod(differencephase_B,2 * pi) / 2 / pi * furiournumber_B) + 1;
differencephase_C_mapped = floor(mod(differencephase_C,2 * pi) / 2 / pi * furiournumber_C) + 1;
differencephase_D_mapped = floor(mod(differencephase_D,2 * pi) / 2 / pi * furiournumber_D) + 1;

figure;
imagesc(differencephase_C_mapped)
colormap(othercolor('RdBu5'))
figure;
imagesc(differencephase_D_mapped)
colormap(othercolor('RdBu5'))
%% 
imagetarget = zeros(furiournumber_A,furiournumber_B,furiournumber_C,furiournumber_D);

imagetarget(13 + 3,7 + 1, 7 + 0, 7 + 1) = 1.18;
% 
imagetarget(13 + 3,7 + 3, 7 + 0, 7 + 0) = 1.2;

imagetarget(13 + 5,7 + 1, 7 + 3, 7 + 1) = 1.1;

imagetarget(13 + 5,7 + 3, 7 + 2, 7 + 3) = 0.9;

imagetargetESPR = imagetarget(13:13+7,7:7+3,7:7+3,7:7+3);
mappingSpace = Fx_PredictionMAT(imagetargetESPR);
mappingSpace = circshift(mappingSpace,4,1);
mappingSpace = circshift(mappingSpace,4,4);
mappingSpace = circshift(mappingSpace,4,11);
%%
finalphase = Fx_Mapping4D(mappingSpace, differencephase_A_mapped, differencephase_B_mapped, differencephase_C_mapped, differencephase_D_mapped, N);
finalphase = finalphase .* exp(1i * 1 * r);
figure;
imagesc(angle(finalphase))
colormap(othercolor('RdBu5'))

%%
Uin = Fx_gaussianbeam(N,N,4,pixel);
Uin(r>6) = 0;

ratio = 0.2;
Uf1 = Fx_CZT(Uin .* finalphase,ratio,N);
figure;
imagesc(abs(Uf1.^1.5));
colormap(turbo)

comp = pi * linspace(1,N,N);
[compx,compy] = meshgrid(comp,comp);
gratingComp = exp(1i * 0.999 * ratio * (compx + compy));
Uf1 = Uf1 .* gratingComp;
figure;
imagesc(angle(Uf1))
Fx_SIM2EXP(finalphase,'snowflake.bmp')

