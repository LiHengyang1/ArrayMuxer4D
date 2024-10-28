function finalphase = Fx_Mapping4D(mappingSpace, differencephase_A_mapped, differencephase_B_mapped, differencephase_C_mapped, differencephase_D_mapped, N)

%% CodeCandy
% tic
% finalphase = arrayfun(@(x,y,z) mappingSpace1(x,y,z),differencephase_D1_mapped,differencephase_D2_mapped,differencephase_D3_mapped);
% toc

%% ForLoop
finalphase = zeros(N);
tic
for ii = 1:N
    for jj = 1:N
        finalphase(ii,jj) = mappingSpace(differencephase_A_mapped(ii,jj), differencephase_B_mapped(ii,jj), differencephase_C_mapped(ii,jj), differencephase_D_mapped(ii,jj));
    end
end
toc

finalphase = exp(1i * finalphase * 2 * pi);
end