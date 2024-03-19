%%
rat_behav_  = load('rat_behav.mat');
rat_behav   = rat_behav_.my_matrix;

rat_neural_ = load('rat_neural.mat');
rat_neural  = rat_neural_.my_matrix;
%%
rat_data    = func0(rat_neural,rat_behav);
params      = CEBRA_defaultParams_rat();
input_directory = '/home/donnarumma/TESTS/CEBRA/RAT/';
output_directory = '/home/donnarumma/TESTS/CEBRA/RAT/';
input_directory = '/home/donnarumma/tools/Cebra_for_all/cebra_codes/';
output_directory = '/home/donnarumma/tools/Cebra_for_all/cebra_codes/';

[rat_data,out]  = cebraCompute(rat_data,params,input_directory,output_directory);

