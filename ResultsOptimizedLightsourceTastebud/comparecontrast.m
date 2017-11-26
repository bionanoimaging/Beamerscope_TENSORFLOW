a = readim();
b = a;
b(a==100)=200;
c = readim('orka');

mutualinformation(a,b)
mutualinformation(a,noise(c,'poisson'))

my_path = '/Users/Bene/Dropbox/Dokumente/Promotion/PROJECTS/Beamerscope/Dokumente/Comparison_Cell/'
my_BF = 'BF.jpg';
my_illopt = 'Illopt.jpg';
my_dpc = 'dpc.jpg';
my_qdpc = 'qdpc.jpg';

bf = readim([my_path, my_BF]);
illopt = readim([my_path, my_illopt]);
dpc = readim([my_path, my_dpc]);
qdpc = readim([my_path, my_qdpc]);



% calculate fidelity BF
mean(abssqr(normminmax(bf)- normminmax(qdpc)))

% calculate fidelity illopt
mean(abssqr(normminmax(illopt)- normminmax(qdpc)))

% calculate fidelity dpc
mean(abssqr(normminmax(dpc)- normminmax(qdpc)))

%%%
% calculate MSE BF
mse(normminmax(bf), normminmax(qdpc))

% calculate MSE illopt
mse(normminmax(illopt), normminmax(qdpc))

% calculate MSE dpc
mse(normminmax(dpc), normminmax(qdpc))


%%%
% calculate mutualinformation BF
mutualinformation(normminmax(bf), normminmax(qdpc))

% calculate mutualinformation illopt
mutualinformation(normminmax(illopt), normminmax(qdpc))

% calculate mutualinformation dpc
mutualinformation(normminmax(dpc), normminmax(qdpc))


%%%
% calculate psnr BF
psnr(normminmax(bf))

% calculate psnr illopt
psnr(normminmax(illopt))

% calculate psnr dpc
psnr(normminmax(dpc))



