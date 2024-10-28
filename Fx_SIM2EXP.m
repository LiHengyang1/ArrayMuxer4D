function Fx_SIM2EXP(slm,nameofslm)

N = length(slm);
H = 1273;
L = 1025;
xshift = 0;
yshift = 0;
xx = linspace(-512,512,L);
yy = linspace(-636,636,H);
[xx,yy] = meshgrid(xx,yy);
[theta,r] = cart2pol(xx,yy);

blocking = ones(H,L);
blocking(r > 0.5 * N) = 0;
grating = exp(1i * 1.2 * (xx + yy));
grating = (1-blocking) .* grating;
grating = angle(grating);

slm0 = zeros(H,L);
slm0(636 - N/2+1:636 + N/2,512 - N/2+1:512 + N/2) = angle(slm);
slm0 = slm0 .* blocking;
slm0 = slm0 + grating;
slm0 = (slm0 + pi) / 2 / pi * 256;
slm0 = floor(slm0);
slm0 = uint8(slm0);
figure;
imagesc((slm0.'))
slm0 = slm0.';
imwrite(slm0,nameofslm);



