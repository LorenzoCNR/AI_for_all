% function TEST_RAT_CEBRA
%%
rat_behav_  = load('rat_behav.mat');
rat_behav   = rat_behav_.my_matrix;

rat_neural_ = load('rat_neural.mat');
rat_neural  = rat_neural_.my_matrix;
%%
rat_data                = arrangeCEBRARatTrials(rat_neural,rat_behav);
params                  = cebraComputeParams();
test_directory          = '/home/donnarumma/TESTS/CEBRA/RAT/';
params.script_name      = '/home/donnarumma/tools/Cebra_for_all/cebra_codes/wrap_py_rat_mod.py';
params.script_out_dir   = test_directory;
params.matlab_out_dir   = test_directory;
[rat_data,out]          = cebraCompute(rat_data,params);

