
my_path = '/Users/Bene/Dropbox/Dokumente/Promotion/PROJECTS/Beamerscope/Dokumente/Comparison_Fiber/'
my_BF0 = 'BF_0.jpg';
my_BF1 = 'BF_1.jpg';
my_BF2 = 'BF_2.jpg';
my_BF3 = 'BF_3.jpg';

my_illopt_1 = 'Illopt_1.jpg';
my_illopt_2 = 'Illopt_2.jpg';
my_dpc = 'dpc.jpg';
my_qdpc = 'qdpc.jpg';
my_df = 'df.jpg';

bf_0 = dip_image(rgb2gray(imread([my_path, my_BF0])));
bf_1 = dip_image(rgb2gray(imread(([my_path, my_BF1]))));
bf_2 = dip_image(rgb2gray(imread(([my_path, my_BF2]))));
bf_3 = dip_image(rgb2gray(imread(([my_path, my_BF3]))));

df = dip_image(rgb2gray(imread(([my_path, my_df]))));

illopt_1 = dip_image(rgb2gray(imread(([my_path, my_illopt_1]))));
illopt_2 = dip_image(rgb2gray(imread(([my_path, my_illopt_2]))));

dpc = dip_image(rgb2gray(imread(([my_path, my_dpc]))));
qdpc = dip_image(rgb2gray(imread(([my_path, my_qdpc]))));



disp('calculate fidelity BF')
mean(abssqr(normminmax(bf_0)- normminmax(qdpc)))

disp('calculate fidelity illopt')
mean(abssqr(normminmax(illopt_1)- normminmax(qdpc)))

disp('calculate fidelity dpc')
mean(abssqr(normminmax(dpc)- normminmax(qdpc)))

%%%
disp('calculate MSE BF')
mse(normminmax(bf_0), normminmax(qdpc))
mse(normminmax(bf_1), normminmax(qdpc))
mse(normminmax(bf_2), normminmax(qdpc))
mse(normminmax(bf_3), normminmax(qdpc))


disp('calculate MSE illopt')
mse(normminmax(illopt_1), normminmax(qdpc))
mse(normminmax(illopt_2), normminmax(qdpc))

disp('calculate MSE dpc')
mse(normminmax(dpc), normminmax(qdpc))

disp('calculate MSE df')
mse(normminmax(df), normminmax(qdpc))

%%%
disp('calculate mutualinformation BF')
mutualinformation(normminmax(bf_0), normminmax(qdpc))

disp('calculate mutualinformation illopt')
mutualinformation(normminmax(illopt_1), normminmax(qdpc))

disp('calculate mutualinformation dpc')
mutualinformation(normminmax(dpc), normminmax(qdpc))

disp('calculate mutualinformation df')
mutualinformation(normminmax(df), normminmax(qdpc))


%%%
disp('calculate psnr BF')
psnr(normminmax(bf_0))

disp('calculate psnr illopt')
psnr(normminmax(illopt_1))

disp('calculate psnr dpc')
psnr(normminmax(dpc))

disp('calculate psnr df')
psnr(normminmax(df))


