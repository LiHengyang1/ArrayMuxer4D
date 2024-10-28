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

delta_A = 14 * lambda * f / L0 / 1;
delta_B = delta_A * 1;
delta_C = 0.2;
delta_D = 0.4;
differencephase_A = k * x / z * delta_A;
differencephase_B = k * y / z * delta_B;
differencephase_C = delta_C * x.^3;
differencephase_D = delta_C * y.^3;
differencephase_A_mapped = floor(mod(differencephase_A,2 * pi) / 2 / pi * furiournumber_A) + 1;
differencephase_B_mapped = floor(mod(differencephase_B,2 * pi) / 2 / pi * furiournumber_B) + 1;
differencephase_C_mapped = floor(mod(differencephase_C,2 * pi) / 2 / pi * furiournumber_C) + 1;
differencephase_D_mapped = floor(mod(differencephase_D,2 * pi) / 2 / pi * furiournumber_D) + 1;
%% 
imagetarget = zeros(furiournumber_A,furiournumber_B,furiournumber_C,furiournumber_D);
imagetarget(13 + 3,7 + 0, 7 + 0, 7 + 1) = 1.0;
imagetarget(13 + 5,7 + 1, 7 + 3, 7 + 0) = 1.0;
imagetarget(13 + 2,7 + 2, 7 + 0, 7 + 3) = 1;
imagetarget(13 + 4,7 + 3, 7 + 3, 7 + 3) = 1; 

imagetargetESPR = imagetarget(13:13+7,7:7+3,7:7+3,7:7+3);
mappingSpace = Fx_PredictionMAT(imagetargetESPR);
mappingSpace = circshift(mappingSpace,4,2);
mappingSpace = circshift(mappingSpace,4,3);
mappingSpace = circshift(mappingSpace,4,4);
%%
finalphase = Fx_Mapping4D(mappingSpace, differencephase_A_mapped, differencephase_B_mapped, differencephase_C_mapped, differencephase_D_mapped, N);
figure;
imagesc(angle(finalphase))
%%
Uin = Fx_gaussianbeam(N,N,4,pixel);
Uin(r>6) = 0;

ratio = 0.2;
Uf1 = Fx_CZT(Uin .* finalphase,ratio,N);
figure;
imagesc(abs(Uf1.^2));
colormap(jet)

for ii = 1:50
    z = 400 + 4 * (ii-20);
    U(:,:,ii) = Fx_Lazy_CZT_SFFT(Uin .* finalphase .* exp(-1i * k * r.^2 / 2 / f),N,z,lambda,L0,0.5*L0,0);
end
U3d = abs(U).^2;
figure;
sliceViewer(U3d)
colormap("turbo")
% save U3d.mat U3d

Fx_SIM2EXP(finalphase,'airy.bmp')