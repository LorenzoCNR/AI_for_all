% function TEST_RAT_CEBRA
%% Step 0 - load raw rat data
clear;
%mat_data_dir                    = '/home/donnarumma/tools/Cebra_for_all/cebra_codes/mat_rat_data/';
%mat_data_dir                    = 'C:\Users\zlollo2\Desktop\Strila_20_03_24\CNR neuroscience\cebra_codes\';
%mat_data_dir                    ='F:\CNR neuroscience\cebra_codes\',
mat_data_dir                    ='/media/zlollo/STRILA/CNR neuroscience/cebra_codes/'
mat_data_dir                    ='C:\Users\zlollo2\Desktop\Strila_27_03_24\CNR neuroscience\cebra_codes\'
rat_behav_                      = load([mat_data_dir 'rat_behav.mat']);
rat_behav                       = rat_behav_.my_matrix;
rat_neural_                     = load([mat_data_dir 'rat_neural.mat']);
rat_neural                      = rat_neural_.my_matrix;
%% Step 1 - arrange in toolbox format
rat_data                        = arrangeCEBRARatTrials(rat_neural,rat_behav);
par.cebraCompute                = cebraComputeParams();

%%disp(['Lo script si trova in: ', par.cebraCompute.script_name]);
%disp(['La directory di output MATLAB è: ', par.cebraCompute.matlab_out_dir]);

%test_directory                  = '/home/donnarumma/TESTS/CEBRA/RAT/';
%test_directory                  = 'C:\Users\zlollo2\Desktop\Strila_20_03_24\CNR neuroscience\cebra_codes';
%test_directory                    =  'F:\CNR neuroscience\cebra_codes';
test_directory                    ='/media/zlollo/STRILA/CNR neuroscience/cebra_codes'
test_directory= 'C:\Users\zlollo2\Desktop\Strila_27_03_24\CNR neuroscience\cebra_codes'
%par.cebraCompute.script_name    = '/home/donnarumma/tools/Cebra_for_all/cebra_codes/wrap_py_rat_mod.py';
%par.cebraCompute.script_name    = 'C:\Users\zlollo2\Desktop\Strila_20_03_24\CNR neuroscience\cebra_codes\wrap_py_rat_mod.py';
%par.cebraCompute.script_name     =  'F:\CNR neuroscience\cebra_codes\wrap_py_rat_mod.py';

par.cebraCompute.script_out_dir = test_directory;
par.cebraCompute.matlab_out_dir = test_directory;
disp(par.cebraCompute);
%% Step 2 - perform cebraCompute
[rat_data,out]                  = cebraCompute(rat_data,par.cebraCompute);

%%% Step 3 - Apply Transform - and (re)convert to trial format
par.cebraProject=cebraProjectParams()
par.cebraProject.model_file='./fitted_model.pkl';
[data_trials,out] = cebraProject(rat_data, par.cebraProject)

