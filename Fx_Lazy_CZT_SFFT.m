function [Uf,Parameters] = Fx_Lazy_CZT_SFFT(U0,N,z0,lambda,L0,Limg,ParameterOutput)

ratio = Limg / (N * lambda * z0 / L0);
k = 2 * pi / lambda;
gridbase = ([0 : N - 1] - (N - 1) / 2).';
[U,V] = meshgrid(gridbase,gridbase);
pixel_L0 = L0 / N;
xx0 = U .* pixel_L0;
yy0 = V .* pixel_L0;
Fresnelcore = exp(1i * k / 2 / z0 * (xx0.^2 + yy0.^2));
f2 = U0 .* Fresnelcore;  %S-FFT计算菲涅耳衍射时的傅里叶变换函数
Uf = Fx_CZT(f2,ratio,N);%对N*N点的离散函数f2作FFT计算
gratingComp = exp(1i * (1-1/N) * ratio * pi * (U + V));
Uf = Uf .* gratingComp;        %能量补偿
Uf = Uf * ratio / N;


pixel_img = lambda * z0 / L0 * ratio;
xx1 = (U-0.5) .* pixel_img;
yy1 = (V-0.5) .* pixel_img;
Uf = Uf .* exp(1i * k / 2 / z0 * (xx1.^2 + yy1.^2));

if ParameterOutput
    Parameters.ratio = ratio;
    Parameters.Fresnelcore = Fresnelcore;
    Parameters.gratingComp = gratingComp;
    Parameters.FresnelCompensate = exp(1i * k / 2 / z0 * (xx1.^2 + yy1.^2));
end
end