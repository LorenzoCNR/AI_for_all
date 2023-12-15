% % Cebra è primariamente usato per estrarre fattori latenti
% da serie temporali. Supporta 3 modelli (cfr più sotto)
% % e si basa su un algoritmo di learning self sueprvised che usa il
% il contrastive learning. Di fatto è una tecnica di riduzione della 
% dimensionalità come tSNE UMAP (con migliori risultati secondo il paper) 
% Può anche essere usato su dati non time series. In generale, CEBRA
% è raccomandato per valutare le variazioni nella consistenza dei
% dati neuroscientifici attraverso diverse condizioni, come aree cerebrali,
% cellule, o animali. Questo significa che può aiutare a identificare come
% variano i modelli di attività cerebrale in situazioni diverse.
% Il metodo può essere utilizzato per interpretare o "decodificare" i 
% segnali neurali in modo che sia guidato da ipotesi specifiche,
% Ciò implicando l'utilizzo di CEBRA per testare teorie' ... 
% specifiche su come il cervello elabora le informazioni.
% 
% CEBRA può aiutare nell'esplorazione della struttura e delle relazioni ' ...
% all'interno degli spazi di embedding, e oltre all'esplorazione, CEBRA è' ...
%  utile anche per visualizzare questi spazi e considerare come cambiano 
% nel tempo o in risposta a stimoli diversi.
% CEBRA è stato utilizzato, per applicazioni pratiche tra cui la mappatura
% dello spazio, la decodifica di film naturali e la formulazione di ipotesi
% sulla codifica neurale nei sistemi sensorimotori. Questi esempi sono 
% tratti da un articolo di Schneider, Lee e Mathis del 2023.

%% cambiare con la folder dove vogliamo mettere tutto
main_folder='/home/zlollo/CNR/Cebra_for_all'
main_folder=''

params = struct('main_dir',main_folder, 'mod_arch', 'offset10-model', 'output_dimension', 3 ,...
'temperature', 1, 'max_iter', 1000, 'distance', 'cosine', 'conditional',...
    'time_delta','time_offsets',10);
save('params.mat', 'params');



%%% Cambiare all'occorrenza
%main_folder='set your folder in'
cd(main_folder)

%path_to="F:\CNR neuroscience\........."
%pyenv('ExecutionMode', 'OutOfProcess')


%setenv('LD_LIBRARY_PATH', '/home/zlollo/anaconda3/envs/cebra/lib')
%pyenv('Version', '/home/zlollo/anaconda3/envs/cebra0/bin/python');

%% UBUNTU (shell digitare 'which python'...percorso python in conda ed ambiente relativo)
pyenv('Version', '/home/zlollo/anaconda3/envs/cebra/bin/python', ...
    'ExecutionMode', 'InProcess');
    

py.sys.path


py.importlib.import_module('os');
py.importlib.import_module('sys');
py.print('Hello from Python');
%jl = py.importlib.import_module('joblib');
pytorch = py.importlib.import_module('torch');
[status, cmdout] = system('python -c "import torch; print(torch.__version__)"');
if status == 0
    disp(['PyTorch Version: ', cmdout]);
else
    disp('Error executing Python command');
end
%np = py.importlib.import_module('numpy');
%pd = py.importlib.import_module('pandas');

%%%% PLOT
% 
% 
% figure('Units', 'inches', 'Position', [0 0 12 4]);
% 
% % Plot neural data
% subplot(1, 2, 1);
% imagesc(neuralDataMat(1:5000, :)');
% colormap('gray');
% ylabel('Neuron #');
% xlabel('Time [s]');
% xticks(linspace(0, 5000, 5));
% xticklabels(linspace(0, 0.025 * 5000, 5));
% 
% % Plot position data
% subplot(1, 2, 2);
% scatter(1:5000, behaviorDataMat(1:5000, 1), 1, 'filled', 'MarkerFaceColor', [0.5 0.5 0.5]);
% ylabel('Position [m]');
% xlabel('Time [s]');
% xticks(linspace(0, 5000, 5));
% xticklabels(linspace(0, 0.025 * 5000, 5));
% 
% drawnow;

%%% create now csv files to store data for python
%%csvwrite('neuralData.csv', neuralDataMat);
%%%csvwrite('behaviorData.csv', behaviorDataMat);
%%% o anche
%save('data.mat', 'neuralDataMat', 'behaviorDataMat');
%load('data.mat')

%%% Facciamo girare codice python 
%%% si generano degli output che poi carico qui

%% settiamo i parametri (questi possono cambiare secondo indicazioni 
% a seguir e a piacimento)

params = struct('mod_arch', 'offset10-model', 'output_dimension', 3 ,...
'temperature', 1, 'max_iter', 1000, 'distance', 'cosine', 'conditional',...
    'time_delta','time_offsets',10);
save('params.mat', 'params');

%% Facciamo girare tutto in python e poi carichiamo output (codice a seguire)
system('python hip_models.py');

%%% Una nota sui modelli che si fanno girare: 
%Questi offrono diversi parametri...tipo
% 1) ARCHITETTURA DELLA RETE (default: offset1-model)
% 2) TEMPERATURE: Fattore per cui scalare la similarità: 
%   parametro che scala la somiglianza tra le coppie positive e negative.
%   Regolando questo parametro, è possibile influenzare quanto fortemente 
%   il modello dovrebbe considerare le coppie come simili o dissimili. 
%   Valori più alti di questo fattore di scala portano alla creazione di
%   embedding più "affilati" e concentrati. In altre parole, aumentando 
%   il valore di questo parametro, gli embedding risultanti saranno più
%   distinti l'uno dall'altro per le coppie negative e più simili tra 
%   loro per le coppie positive. Di fatto,  questo aiuta a migliorare la 
%   capacità del modello di distinguere tra diversi tipi di dati
% 3) OUTPUT DEMENSION: la dimensione dello spaio di arrivo (embedding)
% 4) MAX ITERATIONS: va da sè
% 5) CONDITIONAL La distribuzione condizionata da utilizzare per campionare 
%   i campioni positivi cioè simili al campione di riferimento. I campioni 
%   di riferimento e  quelli negativi vengono estratti da una prior uniforme. 
%   In particolare, sono supoortate 3 tipi di distribuzione
%   -time: I campioni positivi sono scelti in base al loro momento temporale,
%   con un offset temporale fisso rispetto ai campioni di riferimento.
%   Questo significa che i campioni positivi sono quelli che si verificano 
%   in un momento specifico prima o dopo il campione di riferimento
%   -time delta: Questo approccio considera come il comportamento 
%   (o le caratteristiche dei dati) cambia nel tempo. 
%   I campioni positivi sono scelti considerando la distribuzione empirica
%   del comportamento o delle caratteristiche all'interno di un intervallo
%   di tempo definito (time_offset).
%   -delta: Qui, i campioni positivi sono scelti in base a una distribuzione
%   gaussiana centrata intorno al campione di riferimento, 
%   con una deviazione standard (delta) fissa. Ciò significa che i 
%   campioni positivi saranno quelli che sono "vicini" al campione di
%   riferimento secondo una misura quantitativa definita dal delta.
%   Campioni di riferimento e negativi: Sia i campioni di riferimento 
%   che quelli negativi (dissimili dal campione di riferimento) vengono 
%   scelti da una distribuzione uniforme, il che significa che vengono 
%   selezionati casualmente dall'intero set di dati senza una preferenza 
%   specifica.
% 6) DEVICE: 
% 7) VERBOSE: Fa vedere come evolve il training 
% 8) TIME_OFFSET:  li "offsets" sono valori che determinano come i
%   campioni vengono selezionati rispetto a un punto di riferimento nel
%   tempo. Questi valori sono cruciali per costruire la distribuzione 
%   empirica, ovvero una rappresentazione basata sui dati effettivi di 
%   come variano le caratteristiche dei campioni nel tempo. L'offset può
%   essere un singolo valore fisso, che significa che tutti i campioni 
%   positivi saranno selezionati con lo stesso intervallo di tempo dal
%   campione di riferimento. Alternativamente, può essere una tupla di
%   valori, da cui il modello campiona uniformemente. Questo permette una
%   maggiore varietà e casualità nella selezione dei campioni positivi, 
%   riflettendo diverse possibili distanze temporali rispetto al campione
%   di riferimento. Time offset ha effetto solo se conditional è
%   settata su "time" o "time_delta"
% 9) HYBRID: se Settata su True, il modello verrà allenato usando funzioni 
%    di perdita che distinguono tra campioni in momenti diversi
%   (time contrastive) e tra campioni che presentano diversi comportamennti
%   o stati. (behavio contrative)
% 10)DISTANCE: funzione di distanza usata nel training per definire i
%    campioni positivi e negativi rispetto ai campioni di riferimento
%    Può essere cosine ed euclidean
    
%%%



%%%% faccio girare i primi modelli cebra in Python 



%%% I tre modelli di cui sopra...
%	HYPOTHESIS (POSITION) Modello SUPERVISED:
    % Addestramento di un modello con output tridimensionale che usa 
    % informazioni posizionali (posizione + direzione).
    % conditional = %time_delta  indica [l'uso di una variabile 'comportamentale
    % ausiliaria (il tempo) durante l']addestramento del modello.
    %Modello SUPERVISED
	% Gli embedding si ottengono modellando i dati neurali ( X, matrice t*m)
    % su i dati di 	posizione come label  (Y, vettore t*1) e applicando 
    % un metodo transform sui dati 	neurali...quello che restituisce è una
    % matrice t*3….3 origina dal fatto che noi stabiliamo un 	output 3d
    

	%DISCOVERY (TIME) 
    %Addestramento di un modello che utilizza solo informazioni temporali, 
    % senza dati comportamentali.
    % conditional = 'time' viene usato per impostare il modello su
    %una modalità che considera solo il tempo.
    % Modello UNSUPERVISED
	%...il modello ed il transform vengono applicati
    % solo ai dati neurali 	(matrice t*n). Anche qui l%ooutput del metodo
    % transform sono gli embedding di dimensione 	t*3 con 3 dimensione
    % output imposta

	%Hybrid...Time + Behaviour:
	%Come POSITION con opzione hybrid...la loss function non usa più solo
    % esempi che si basano 	sul tempo (dati neurali) ma anche su dati
    % comportamentalie 
    % Addestramento di un modello che usa sia informazioni temporali
    %   che posizionali.
    %   hybrid = True indica [l'uso combinato di informazioni temporali ' ...
    %        'e comportamentali.]

%	SHUFFLED
%	Vengono randomizzati i dati direzione posizione (matrice t*3) e poi si 
% fa girare il modello 	con X, dati 	neurali (matrice t*m) sui dati 
% randomizzati. Anche qui gli embedding si 	ottengono applicando un metodo
% transform sui dati neurali. Otteniamo sempre una matrice 	t*3.


behaviorData=load("beavior_data.mat")

%% controllare che quando li metti in matrice numerica abbia dimensione
%% (time)steps*variabili

behav_labels=[behaviorData.dir', behaviorData.left', behaviorData.right']

%%%  salvo in un mat cebra_1step_output

%%% viisualizziamo gli spazi embedded generati (Fig 2b)

%%% le colormap si possono cambiare
cebra_1step_output=load("cebra_1step_output.mat")

plot2b(cebra_1step_output, behav_labels)


%%&%%%% (Fig 2c) TEST IPOTESI:ADDESTRIAMO MODELLI CON DIVERSE IPOTESI %%%%%%
%  sull'encoding posizionale dell'ippocampo. L'obiettivo è confrontare
%  diversi modelli CEBRA-Behavior addestrati utilizzando diverse variabili 
%  comportamentali: solo la posizione, solo la direzione, entrambe queste 
%  variabili, e modelli di controllo con variabili comportamentali mescolate
%  casualmente (shuffled).
% Si utilizza  quindi una dimensione del modello predefinita. 
% Nel lavoro originale descritto nel documento, sono state utilizzate 
% dimensioni del modello che variano da 3 a 64 per analizzare i dati
% dell'ippocampo, e si è  osservata una topologia coerente attraverso 
% queste diverse dimensioni.
% Per le analisi di decodifica successive, si utilizzerà un set di dati
% diviso, con l'80% dei dati train e il 20% test. 
% Solo il train set sarà  usato per addestrare i modelli.
% In sintesi, lo scopo di questo esperimento è di esplorare come differenti 
% tipi di variabili comportamentali (posizione, direzione, entrambe, 
% o mescolate) influenzino la capacità dei modelli CEBRA-Behavior di
% codificare informazioni sull'ippocampo. 
% 
cebra_2nd_output_hyp_test=load("cebra_2nd_output_hyp_test.mat")
plot2c(cebra_2nd_output_hyp_test, behav_labels)

%%%%% PLOTTIAMO LA PERDITA DEI MODELLI fig 2c_loss 
%InfoNCE è una funzione di perdita utilizzata in contesti di apprendimento
% di rappresentazioni, specialmente in tecniche di apprendimento
% contrastivo. Misura la perdita (errore) tra le rappresentazioni 
% positive (simili) e negative (dissimili).
% Una tendenza decrescente nella perdita InfoNCE suggerirebbe che il 
% modello sta migliorando nella sua capacità di distinguere tra esempi 
% positivi e negativi nel contesto dell'apprendimento contrastivo.

models_loss=load("model_loss.mat");

loss_pos_dir = models_loss.loss_pos_dir;
loss_pos = models_loss.loss_pos;
loss_dir = models_loss.loss_dir;
loss_pos_dir_shuffle = models_loss.loss_pos_dir_shuffle;
loss_pos_shuffle = models_loss.loss_pos_shuffle;
loss_dir_shuffle =models_loss.loss_dir_shuffle;

% Plot delle perdite dei modelli
fig = figure;
ax = axes(fig);
plot(ax, loss_pos_dir, 'Color', '#00BFFF', 'DisplayName', 'position+direction'); 
hold(ax, 'on'); % Mantiene il grafico corrente per sovrapporre le altre linee
plot(ax, loss_pos, 'Color', [0 191/255 1 0.3], 'DisplayName', 'position');
plot(ax, loss_dir, 'Color', [0 191/255 1 0.6], 'DisplayName', 'direction'); 
plot(ax, loss_pos_dir_shuffle, 'Color', '#808080', 'DisplayName', 'pos+dir, shuffled');
plot(ax, loss_pos_shuffle, 'Color', [128/255 128/255 128/255 0.3], 'DisplayName', 'position, shuffled'); % grigio più chiaro
plot(ax, loss_dir_shuffle, 'Color', [128/255 128/255 128/255 0.6], 'DisplayName', 'direction, shuffled'); % grigio più chiaro


legend(ax, 'show');
xlabel(ax, 'Iterations');
ylabel(ax, 'InfoCNE Loss ');
title(ax, 'Model Loss Comparison');

hold(ax, 'off');

%% fig 2d
% Valutiamo la capacità di decoding e di ricostruzione di informazioni
% dalle rappresentazioni codificate e le caratteristiche estratte 
% SIamo in particolare interessati alla decodifica di info relative a 
% posizione e direzione dagli embedding del ns modello. 
% Si usa il KNN. regressor per decodificare la posizione e il classifier
% per decodificare la direzione 
% si scelgono 36 punti secondo un cosine metric. La preisione della
% decodifica viene valutata con R2 e l'errore mediano assoluto.
% sostanzialmente si usano gli output dei modelli precedenti

models_decode=load("cebra_decoding.mat");
start_idx = 320
llength = 700
history_len = 700
framerate = 25 / 1000
linewidth = 2
fig2d(models_decode, start_idx, framerate, llength ,linewidth)

%% fig 2d (sul pre print) Erorre e perdita
% dati per grafico a barre
barData = [models_decode.cebra_posdir_decode, models_decode.cebra_pos_decode, ...
          models_decode.cebra_dir_decode, models_decode.cebra_posdir_shuffled_decode, ...
           models_decode.cebra_pos_shuffled_decode, models_decode.cebra_dir_shuffled_decode];

%  grafico a barre
figure;
subplot(1, 2, 1); % Primo subplot
bar(barData, 'FaceColor', [0.5, 0.5, 0.5]); 
set(gca, 'XTickLabel', {'pos+dir', 'pos', 'dir', 'pos+dir, shuffled', ...
                         'pos, shuffled', 'dir, shuffled'});
ylabel('Median err. [m]');

%% dati grfico scatter

% dati di perdita 
lossData = [models_decode.cebra_posdir_loss(end), models_decode.cebra_pos_loss(end), ...
            models_decode.cebra_dir_loss(end),models_decode.cebra_posdir_shuffled_loss(end), ...
            models_decode.cebra_pos_shuffled_loss(end), models_decode.cebra_dir_shuffled_loss(end)];

%dati di errore di decodifica
scatterResults = [models_decode.cebra_posdir_decode, models_decode.cebra_pos_decode, ...
                  models_decode.cebra_dir_decode, models_decode.cebra_posdir_shuffled_decode, ...
                  models_decode.cebra_pos_shuffled_decode, models_decode.cebra_dir_shuffled_decode];

subplot(1, 2, 2); 
hold on;
scatter(models_decode.cebra_posdir_loss(end), models_decode.cebra_posdir_decode, 'filled', 'DisplayName', 'position+direction');
scatter(models_decode.cebra_pos_loss(end), models_decode.cebra_pos_decode, 'filled', 'DisplayName', 'position_only');
scatter(models_decode.cebra_dir_loss(end), models_decode.cebra_dir_decode, 'filled', 'DisplayName', 'direction_only');
scatter(models_decode.cebra_posdir_shuffled_loss(end), models_decode.cebra_posdir_shuffled_decode, 'filled', 'DisplayName', 'pos+dir, shuffled');
scatter(models_decode.cebra_pos_shuffled_loss(end), models_decode.cebra_pos_shuffled_decode, 'filled', 'DisplayName', 'position, shuffled');
scatter(models_decode.cebra_dir_shuffled_loss(end), models_decode.cebra_dir_shuffled_decode, 'filled', 'DisplayName', 'direction, shuffled');
hold off; 
xlabel('InfoNCE Loss');
ylabel('Decoding Median Err.');
legend('Location', 'eastoutside'); 


