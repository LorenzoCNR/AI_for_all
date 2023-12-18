
% Imposto ambiente python per farlo girare da MatLab
% UBUNTU (shell digitare 'which python'...percorso python in conda ed ambiente relativo)
pyenv('Version', '/home/zlollo/anaconda3/envs/cebra/bin/python', ...
    'ExecutionMode', 'InProcess');
    

%%% Aggiungere le path di interesse dove in particolare scarico  i dati
% (ricordarsi di farlo anche nel file python wrap_py.py

main_folder="/home/zlollo/CNR/git_out_cebra/elab_Mirco";
cd(main_folder)
%save('path_to_save.mat', 'main_folder', '-v7');


%%% Carico dati:
%%% N.B.! la struttura è semepre nella forma:
% trials: tre liste che includono l'id, le matrici con i dati per ongi
% trial; le label corrispondenti (in forma numerica)
% labels: valori categorici (o qualitativi) assunti dalle labels. 

data_norm_=load("data_Baseline_NormSingle.mat");
data_norm=data_norm_.dataFITTS


%%% Estraggo i dati e li metto in un formato idoneo per CEBRA
% ottengo un mat che poi carico in python per le elaborazioni
data=data_norm.trials;
fields = fieldnames(data);
labels=cell2mat({data(1:end).trialType})

numElectrodes = size(data(1).EEG,1);
numTimePoints = size(data(2).EEG,2);
numMatrices = size(data,2) % Numero totale di matrici
data_norm_long = []; % Inizializza una matrice vuota


%%% faccio Re-Shape e unisco tutte le matrici (359 nello specifico)
for i = 1:numMatrices
    % Estraggo l'i-esima matrice e metto in formato tempo-elettrodi
     currentMatrix = data(i).EEG'; 
     currentLabel = labels(i);

    %[electrodeGrid, timeGrid] = meshgrid( 1:numTimePoints,1:numElectrodes,);

    % Aggiungi un identificatore di matrice (label)
    matrixID = repmat(currentLabel,size(currentMatrix, 1), 1);

    % Concatena con il dataset complessivo
    data_norm_long= [data_norm_long; [currentMatrix,matrixID]];
end

save("data_norm_long.mat")

%%%% Parametri per il modello 
% model type supervised/unsupervised a seconda che voglia o meno
%%% CFR i link sotto per la definizione dei parametri
% https://cebra.ai/docs/api/pytorch/models.html#cebra.models.get_options
% https://cebra.ai/docs/api/sklearn/cebra.html#cebra.CEBRA.temperature_mode

%model_architecture='offset1-model', DA CAPIRE cfr guida in link sopra
% criterion='infonce' unica disponibile, NON in model_params
% device='cuda_if_available' Unica disponibile, NON in model_params
% distance='cosine', ('euclidean') 
% conditional=None time_delta,time, delta (3 types of distribution) 
% temperature=1.0,
% temperature_mode='constant', fase sperimentale,NON in model_params
% min_temperature=0.1,  sperimentale, NON in model_params
% time_offsets=1, ha effetto solo se conditional=time o delta
% delta=None, non si capisce, NON in model_params
% max_iterations=10000, 
% max_adapt_iterations=500, attivo solo se in cebra.CEBRA.fit() adapt=True 
% batch_size=512
% learning_rate=0.0003,
% optimizer='adam', unico supportato, NON in model_params
% output_dimension=8,
% verbose=False, 
% num_hidden_units=32,
% pad_before_transform=True,
% hybrid=False, (se setti True, conditional ='time_delta'
% optimizer_kwargs=(('betas', (0.9, 0.999)), ('eps', 1e-08), ('weight_decay', 0), ('amsgrad', False)))



%%%$ N.B.!!! 
% modello Unsupervised (discovery-time) vuole param conditonal='time'
% modello supervised (behaviour) conditonal come vogliamo
% hybrid = True (supervised) conditional ='time_delta' 
% questione batch size...se ci sono problemi di memoria, diminuire 
% batch size (256 vengono gestiti da una gpu con 12gb nel caso di 
%% modello più complesso, supervised+hybrid)
%% 
params = struct('model_type','supervised','mod_arch', 'offset10-model', ...
    'distance', 'cosine', 'conditional', 'time_delta', 'temperature', 1, 'time_offsets', 10, ...
    'max_iter', 1000, 'max_adapt_iter', 500, 'batch_size', 256, 'learning_rate', 3e-4, ...
    'output_dimension', 3, 'verbose', 'True', 'num_hidden_units', 32, ...
    'pad_before_transform', 'True', 'hybrid', 'false');

save('params.mat', 'params');



%% Facciamo girare tutto in python e poi carichiamo output (codice a seguire)
system('python wrap_py.py');


%%% Carico output Cebra per elaborazioni
load("cebra_output.mat")

