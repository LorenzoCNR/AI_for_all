% function TEST_RAT_CEBRA
%% Step 0 - load raw rat data
clear;
mat_data_dir                    = '/home/donnarumma/tools/Cebra_for_all/cebra_codes/mat_rat_data/';
rat_behav_                      = load([mat_data_dir 'rat_behav.mat']);
rat_behav                       = rat_behav_.my_matrix;
rat_neural_                     = load([mat_data_dir 'rat_neural.mat']);
rat_neural                      = rat_neural_.my_matrix;
%% Step 1 - arrange in toolbox format
rat_data                        = arrangeCEBRARatTrials(rat_neural,rat_behav);
par.cebraCompute                = cebraComputeParams();
test_directory                  = '/home/donnarumma/TESTS/CEBRA/RAT/';
par.cebraCompute.script_name    = '/home/donnarumma/tools/Cebra_for_all/cebra_codes/wrap_py_rat_mod.py';
par.cebraCompute.script_out_dir = test_directory;
par.cebraCompute.matlab_out_dir = test_directory;
disp(par.cebraCompute);
%% Step 2 - perform cebraCompute
[rat_data,out]                  = cebraCompute(rat_data,par.cebraCompute);

