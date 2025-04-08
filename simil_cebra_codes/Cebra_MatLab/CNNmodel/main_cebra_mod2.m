%%%% Aggiungere le path di interesse
%cd 
%$main_folder="/home/zlollo/CNR/git_out_cebra/"
main_folder='/media/zlollo/STRILA/CNR neuroscience/'
cd(main_folder)
%addpath("/home/zlollo/CNR/git_out_cebra/elab_Mirco")

input_directory= '/media/zlollo/STRILA/CNR neuroscience/'
output_directory= '/media/zlollo/STRILA/CNR neuroscience/'
system('python --version')


%% carico dati  python 
%% okkio a come chiama oggetti interni alla struttura (qui è my_matrix)

rat_behav_=load('rat_behav.mat')
rat_behav=rat_behav_.my_matrix

rat_neural_=load('rat_neural.mat')
rat_neur=rat_neural_.my_matrix

%% func0 creo la struttura con dati in trial
%% spikes, labels (behav), time
rat_data = func0(rat_neur,rat_behav)

%% funct1 ha come input i dati in trial, (iper)prametri, le directory
params=CEBRA_defaultParams_rat();
model_w = func1(rat_data,params,input_directory,output_directory)


% optimizer_kwargs=(('betas', (0.9, 0.999)), ('eps', 1e-08), ('weight_decay', 0), ('amsgrad', False)))

%%%% Func2 sarebbe conv1d_layer che  mi calcola i layer sequenziali 
%%% dati i parametri ottenuti in func1

%model_w=load('model_struct.mat')

%%% I layer (ingresso rete)
pad_size=1
%ww_1st=permute(model_w.net_0_weight,[1,3,2])
ww_1st=model_w.net_0_weight,[1,3,2]
bb_1st=model_w.net_0_bias
x_in_1st=rat_neur
x_out_1st = conv1d_layer(x_in_1st, ww_1st, bb_1st, pad_size, ...
    'activation_fn', @gelu, 'use_skip_connection',false );


%%% II layer 
ww_2nd=model_w.net_2_module_0_weight
bb_2nd=model_w.net_2_module_0_bias
x_in_2nd=x_out_1st
x_out_2nd = conv1d_layer(x_in_2nd, ww_2nd, bb_2nd, pad_size, ...
    'activation_fn', @gelu, 'use_skip_connection', true);

%%% III layer 
ww_3rd=model_w.net_3_module_0_weight
bb_3rd=model_w.net_3_module_0_bias
x_in_3rd=x_out_2nd

x_out_3rd = conv1d_layer(x_in_3rd, ww_3rd, bb_3rd, pad_size, ...
    'activation_fn', @gelu, 'use_skip_connection', true);

%%% IV layer 
ww_4th=model_w.net_4_module_0_weight
bb_4th=model_w.net_4_module_0_bias
x_in_4th=x_out_3rd

x_out_4th = conv1d_layer(x_in_4th, ww_4th, bb_4th, pad_size, ...
    'activation_fn', @gelu, 'use_skip_connection', true,'normalize_output' ...
    , false);

%%% V layer  (uscita)
ww_5th=model_w.net_5_weight
bb_5th=model_w.net_5_bias
x_in_5th=x_out_4th

x_out_5th = conv1d_layer(x_in_5th, ww_5th, bb_5th, pad_size, ...
 'activation_fn', @(x) x, 'use_skip_connection', false,'normalize_output' ...
    , true);



%plot2b(cebra_1step_output, behav_labels)




