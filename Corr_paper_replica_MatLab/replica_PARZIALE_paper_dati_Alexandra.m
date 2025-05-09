%%% Ref libreria python https://zenodo.org/records/12359299
%%% parametrizzare blocchi
%% label temporale
%%% distinzione 



% quando scrivo "per loro", intendo per Hasson e compagnia cantante

% Replica moooolto semplificata del paper
% mi sono focalizzato sui punti chiave; 
% 1) usiamo i dati s e k delle scimmie nelle reciproche condizioni 1 e 2;
%    tutta la parte di allineamento dei dati (dialoghi e dati neurali nel
%    loro caso, movimenti del braccio e dati neurali nel nostro, sono
%    saltati.
% 2) I dati sono ricampionati e ridotti a un quinto della lunghezza
%     orginaria (formula K=[(N-W)/S]+1...per ogni trial
%     K Numero di punti rimanenti (118 obs (una per 5 ms) per un totale di 599 ms)
%     N lunghezza dati originari (599 obs (una per ms) obs per trial), 
%     W(indow) ampiezza finestra (10), 
%     S(hift) passo di avanzamento (5) 
%     

% 3)  Genero gli embeddidng con CEBRA behaviour (diciamo supervised)
%     usando più o meno sempre la stessa struttura: 
%     batch_size: 1024
%     max_iterations: 8000-12000
%     hybrid: False
%     verbose: True
%     time_offsets: 10
%     output_dimension: 3
%     learning_rate: 0.0001
%     num_hidden_units: 32-64
%     temperature: 1

% Gli embedding sono il corrispettivo di quello che loro generano con le
% parole usando gpt2. Dati proiettati su uno spazio diverso.
% Il fatto che ci siano parole (movimenti) ripetuti non rileva in quanto si
% tratta (in teoria) di manifold contestuali

% 2) ho allineato i dati campionando come lo fanno loro
% 3) regressione ridge per lag e per elettrodo con cross validation sui dati 
%    e calcolo correlazioni tra valori predetti e osservati -(correlazione
%    tra segmenti dello stimolo)
% 4) Brain to brain coupling 
%  praticamente testano il modello allenato su un soggetto, sull'altro
%  soggetto...

%%% Le parti di riferimento sul paper sono nei metodi del paper:
% "A shared model-based linguistic space for transmitting our thoughts from brain to brain in natural conversations"
% Encoding analysis
% Model-based brain-to-brain coupling
%%% CARICAMENTO DATI %%%
load("cebra_results.mat")

%%  DATI Neuralii %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 64 trials da 599 ms (e una osservazione per ms) ricampionati con finestra
% ovelapping di 10 ms e passo 5 
n_trial=64; 
% lunghezza trial originaria 
trial_length=599
% valori usati per resampling
shift_=5 
window_=10 
% lunghezza trial resamplati
trial_len_res=floor((trial_length-window_)/shift_)+1
n_label=8;

% frequenza campionamento (numero di punti al secondo...dopo aver
% ricampionato abbiamo un quinto dei dati circa)
%  % per loro in hertz sono 512 (punit al secondo)
fs_=round(1000/(trial_length/trial_len_res));

% durata sessione in secondi (64 trials da 599 ms/bin...ridotti a 118 bin)
% circa 38" totali
dd= n_trial*trial_length/1000; % per loro erano 60" di dialogo
% tempo totale in bin o momenti tempi finali
tt_single = (0:trial_len_res-1) * shift_ + window_;  %  un singolo trial
tt = zeros(1, n_trial * trial_len_res);

for i = 1:n_trial
    start_idx = (i-1) * trial_len_res + 1;
    stop_idx  = i * trial_len_res;
    trial_start = (i-1) * trial_length;  % in ms
    tt(start_idx:stop_idx) = tt_single + trial_start;
end     
% tempo in secondi
tt = tt / 1000;  % 

% numero di azioni (per loro erano le parole nell'intervallo di tempo - 1 minuto -)
% per noi è il numero di gesti (8) divisi in trial (di fatto è il numero di
% trial dacchè sono 8 direzioni ripetute per 8 volte (per sessione)

moves_=n_trial;

%%% Prendo i dati neurali in condizione 1 per k e in condizione 2 per S
%%% che dovrebbe essere K si muove e  S guarda K muoversi
% speaker (nel nostro setting è la scimmia K che muove il braccio...
%  condizione 1)

k_active_neural_cond1=k_cond_1_neural_active;

%listener (nel nostro setting immagino sia la scimmia S che guarda K muovere
% il braccio... condizione 2 quindi)
s_passive_neural_cond1=s_cond_1_neural_active;

% numero elettrodi....da capire come usare e adattare questa info poichè 
% nel mio   caso non sono lo stesso numero
%n_channels=;

% onset dei movimenti (nel loro caso si considera un intorno di durata arbitraria
% tipo -+4 sec per ogni parola lungo la durata complessiva del dialogo) 
% nel nostro setting abbiamo un gesto che inizia (in ogni trial)
% 100 ms dal principio del trial. Inoltre, invece che considerare un
% intorno del movimento consideriamo solo momenti successivi (ref. Fra)
% considerando 64 trial di 118 momenti (5ms) so che il movimento inizia
% alla fine della 19ma obs per trial (91-100 ms).

%intorno_sec=4;
%word_onsets = sort(randi([intorno_sec/2*fs length(tt)-intorno_sec/2*fs], 1, wpm));

%%% devo trovare la finestra dove cade l'onset del movimento (per trial)
% la 19ma finestra
onset_bin=find(tt>=0.1,1)
mov_onsets=tt(onset_bin:trial_len_res:end)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% embeddings MANIFOLD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% - per loro sono generati con GPT2 (che per ogni parola tira fuori un 
% embeddding di 1600 componenti)...
% per noi sono quelli di cebra qui metto meno componenti

% embedding_size = 500; 
embeddings_k_active_cond1 = k_cond_1_embed
embeddings_s_passive_cond1 = s_cond_1_embed

%% ALLINEAMENTO DATI NEURALI - DATI movimento Embedding %%%%%%%%%%%%%%%%
% di fatto si analizza l'attività neurale in intervalli di tempo specifici
% successivi al principio del movimento (coem detto, loro lo fanno intorno
% agli onsets delle parole  4 secondi prima e 4 dopo ogni parola). prendiamo
% fino a k  momenti successivi  con k entro la lungheza del trial:
% onset_bin<k<=trial_len_res 

% come loro, che creano finestre di 250ms con overlap di 62.5 ms, creiamo
% finestre con overlap; in particolare,  considero finestre di 200 ms 
% (quindi 40 bins/momenti dato il resampling)  con overlap di 40 ms (8 bin)
% ...da cui, per ogni trial, considerando  che l'evneto avviene al 19mo bin 
% creo finestre 20-59, 28-67 e così via per un numero arbitrario di blocchi 
% (entro la fine del trial)

% Analogamente procediamo per manifold generate con CEBRA.
% questo è altro elemnto distintivo dal paper in quanto loro hanno un
% embedding comune generato con GPT (oltretutto su uno spazio molto più
% grande) mentre noi abbiamo 2 manifold (1 per scimmia) su 3 dimensioni

% inoltre, il loro embedding è statico...il nostro si muove nel tempo; per
% loro ogni parola ha un suo embedding multidimensionale nello spazio ma
% unidimensionale nel tempo; i nostri embedding si muovono nel tempo e
% nello spazio (per avere una situazione analoga alla loro dovrei avere per
% ogni movimento una solo osservazione nello spazio 3d). Da un lato questo
% torna comodo in quanto alla fine mi ritrovo con strutture analoghe per
% dati neurali ed embedding

% sono bin da 5 ms 
% da cui se block size è 20 e shift block 10, sono blocchi da 100 ms con
% overlap di 50 ms (ovviamente successivi all'onset del movimento)
% 1-100ms, 51-150ms
% ....
block_size=20;
shift_block=4 ;

%max_blocks_per_trial_= floor((trial_len_res-n_onset-block_size)/shift_block)+1

for i = 1:n_trial
    onset_pos=find(tt==mov_onsets(i))
    % momento immediatamente successivo onset iniziale, specificamente
    % 5 millisecondi dopo
    idx_min =onset_pos+1 
    % 
    idx_max = onset_pos+trial_len_res-onset_bin
    start_idx = idx_min;
    blocchi_idx = []; 
    
    while start_idx + block_size - 1 <= idx_max
        end_idx = start_idx + block_size - 1;
        blocchi_idx = [blocchi_idx; start_idx, end_idx];
        
        % Calcola il prossimo indice di partenza con shift
        start_idx = start_idx + shift_block;
    end

    %  alla fine mi ritrovo con due strutture (per K ed S)
    % che rappresentano l'attività neurale dei soggetti con riferiemnto 
    % l'onset del movimento; avremo un tot di  blocchi per ogni movimento.
    % con blocco intendo registrazioni neurali successive all'onset del
    % movimento secondo lo schema (con sovrapposizione) indicato in
    % precedenza. (ogni blocco sarà di dimensione 40 bin temporali (200ms
    % quindI per il numero di canali/elettrodi.
    % Ogni blocco rappresenta quindi osservazioni neurali "laggate" nel
    % tempo

    % per quanto riguarda le manifold, da esse prendo i dati
    % "contemporanei" a quelli estratti per le osservazioni neurali
      
    dati_K{i} = cell(size(blocchi_idx, 1), 1);
    dati_S{i} = cell(size(blocchi_idx, 1), 1);
    manif_K_match{i} = cell(size(blocchi_idx, 1), 1);
    manif_S_match{i} = cell(size(blocchi_idx, 1), 1);
    for j = 1:size(blocchi_idx, 1)
        start_blocco = blocchi_idx(j, 1);
        end_blocco = blocchi_idx(j, 2);
        dati_K{i}{j} = k_active_neural_cond1(start_blocco:end_blocco, :);
        dati_S{i}{j} =s_passive_neural_cond1(start_blocco:end_blocco, :);
        manif_K_match{i}{j}=k_cond_1_embed(start_blocco:end_blocco, :);
        manif_S_match{i}{j}=s_cond_1_embed(start_blocco:end_blocco,:)
    end
end

% quidni per ogni scimmia mi ritrovo con strutture di dimensione 1
% (soggetto, K o S) per 64 (movimenti) e dentro ogni cella ho i dati
% neurali laggati con riferimento il principio del movimento:
% quindi 64 movimenti...ogni movimento ha 8 blocchi di dimensione 
% 40 (200ms in bin) per #neuroni/canali/elettrodi

% per quanto riguarda le manifold anche qui, 64 celle (una per movimento)
% che contengono i dati cebra momenti contemporanei ai dati neurali estratti, dop

%% RIDGE REGRESSION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Nel paper originale, l'embedding (X) è statico per ogni parola/movimento: dimensione 
% [#movimenti × spazio_embedding]
% L'attività neurale (Y) è laggata: per ogni lag e per ogni neurone 
% si fa la media sul tempo del blocco neurale.
% Il risultato è un vettore Y di dimensione [parole × 1] per ogni combinazione
% (neurone × lag). Ad esempio ipotizzando 150 parole, 60 blocchi temporali 
% intorno all'onset di ogni parola e 64 neuroni/canali, avrò 61*64 y di
% dimensione 150*1. La X sarà 150*dimensione embedding
% Si esegue quindi una regressione per ciascun neurone 
% e per ciascun lag,   usando sempre la stessa X.

% Nel nostro caso,  L'attività neurale (Y) ha la stessa struttura: 
% una cella per movimento,; ogni cella contiene blocchi/lag di dimensione 
% [t_lag × #neuroni]
% Tuttavia, anche gli embedding (X) hanno dimensione temporale: ogni X è
% una cella per movimento, contenente [t_lag × dimensione_embedding]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% QUESTO 
% Non volendo perdere la componente temporale degli embedding  invece di 
% fare media temporale su Y e usare X statico, trasformiamo i dati in
% formato long e cioè Per ogni lag e per ogni neurone, costrusico un vettore 
% Y_flattened = reshape(Y, [(t_lag * #movimenti), 1]) E una matrice 
% X_flattened = reshape(X, [t_lag * #movimenti), dimensione_embedding])
% Farò quindi come nel paper una regressione per ogni combinazione 
% (neurone × lag) dove Y è un vettore lungo (tempo × movimenti), e X è 
%  una matrice ((tempo × movimenti) × embedding_dim)


% 1) Costruisco le y e recupero le X 
% Numero di blocchi per movimento
n_blocchi = size(blocchi_idx, 1); 
% canali...per ora li tengo distinti (ognuno con la sua numerosità)
% (nel paper i canali sono gli stessi per speaker e listener)
n_channels_k=202
n_channels_s=184
n_output=3
% UNICO VALORE 
% canali (unica numerosità per entrambi = min(Ch(s.k))
%chns=min(n_channels_s,n_channels_k)

% struttura Y (S e K)
% (saranno tanti vettori di dimensione (numero movimenti*1) 
% un vettore per ogni lag per ogni elettrodo...dal momento che  ogni lag (per
% ogni elettrodo) include un blocco di osservazioni per movimento, si fa una
% media sul blocco...quidni un valore per ogni parola per ogni lag per ogni elettrodo
% Blocchi di variabili di rispsota per soggetto

%%%%%%%%%%%%%%%%%%% SCIMMIA K
Y_K= cell(n_blocchi, n_channels_k);
X_K= cell(n_blocchi,1);
% Y_K 
% Ogni elemento Y_K{b,e} è un vettore lungo di dimensione [t_lag * n_trial × 1]
% ottenuto concatenando nel tempo le osservazioni neurali relative al neurone e
% per ciascun movimento p, nel blocco/lag b.

%%% se voglio i blocchi senza annullare la dimensione temporael devo
%%% togliere il mean all'interno del ciclo quando formo le Y_K
% ad ogni modo, quando faccio media sulla lunghezza del blocco mi ritrvo
% con 64 osservazioni (per blocco per elettrodo) che sono le medie dei
% blocchi per ogni movimento . 
% se d'altra parte non faccio la media, per ogni trial (64) per ogni blocco
% per ogni elettrodo, mi trovo con una colonna di osservazioni che ha come
% ulteriore info la lungheza del blocco che rende il vettore lungo 64
% osservaioni ognuna mpltoiplicata per la lunghezza del blocco (estesa)
% quidni se i blocchi sono da 20 obs il vettore diventa
% (n_trial*len_blocco)*1 (è una struttura longitudinale)

for b = 1:n_blocchi
    for e = 1:n_channels_k
        Y_K{b,e} = [];
        %Y_l{b,e} = [];
        for p = 1:n_trial         
            Y_K{b,e} = [Y_K{b,e}; mean(dati_K{p}{b}(:, e))];
            %Y_l{b,e} = [Y_l{b,e}; mean(blocchi_dati{p}{b}(:, e))];
        end
    end
end

% X_K
for b = 1:n_blocchi
        X_K{b} = [];
        %Y_l{b,e} = [];
        for p = 1:n_trial
            
            X_K{b,1} = [X_K{b,1}; mean(manif_K_match{p}{b})];
            %Y_l{b,e} = [Y_l{b,e}; mean(blocchi_dati{p}{b}(:, e))];
        end
 end

%%%%%%%%%%%%%%% SCIMMIA S
Y_S= cell(n_blocchi, n_channels_s);
X_S= cell(n_blocchi, 1);

% Y_s Riempimento di Y_S con i dati
for b = 1:n_blocchi
    for e = 1:n_channels_s
        Y_S{b,e} = [];
        %Y_l{b,e} = [];
        for p = 1:n_trial
            Y_S{b,e} = [Y_S{b,e}; mean(dati_S{p}{b}(:, e))];
        end
    end
end

% X_S
for b = 1:n_blocchi
    
        X_S{b} = [];
        %Y_l{b,e} = [];
        for p = 1:n_trial
            
            X_S{b,1} = [X_S{b,1}; mean(manif_S_match{p}{b})];
        end
    end

% Alla fine per ogni scimmia mi ritrovo con due strutture Y e X. La Y
% contiene tanti vettori di dimensione #movimenti*1 (se mantenessi il tempo 
% ogni vettore sarebbe espanso in formato long e avrebbe una dimensione 
% t_lag*#movimenti*1)

% In particolare 
% facciamo una regressione per lag e per neurone come nel paper originale
% la X contiene le manifold allineate nel tempo (anche qui ho fatto media 
% sulla lungheza del blocco) per blocco temporale. Quindi
% qui, diversamente dal paper, abbiamo una X che cambia nel tempo . In
% particolare abbiamo una X diversa per ogni blocco. X di dimensione
% (#movimenti)*dimensione output (otuput quando genero l'embedding)
% quindi nel paper abbiamo una X su cui regrediamo 60*64 y (lag*neuroni)
% noi abbiamo una X per blocco di dimensione #trial*output_dim
% e su ognuna regrediamo un numero y pari al numero di neuroni per scimmia


%% 
lambda = 1000; % Supponiamo un lambda già ottimale

% Inizializzazione delle strutture per i coefficienti e predizioni
%  scimmia K
coefs_k = cell(n_blocchi, n_channels_k);
y_hat_k = cell(n_blocchi, n_channels_k);
%  scimmia S
coefs_s = cell(n_blocchi, n_channels_s);
y_hat_s = cell(n_blocchi, n_channels_s);

% ottengo quindi delle y stimate e dei coefficienti (uno per ogni aspetto
% dell'embedding delle parole...quindi avendo generato  embedding di 500
% valori avrò 500 coefficienti per regressione)

%Scimmia k
for b = 1:n_blocchi
    % matrice di regressori (una per blocco)
    X=X_K{b}
    I = eye(size(X,2));
    for k = 1:n_channels_k
        % target (segnale neurale per quel blocco ed elettrodo)
        % sono le y osservate per la scimmia 
        y_k = Y_K{b, k}; 
        %%% check su dimensioni
        %if size(X,1) ~= size(Y_target,1)
         %   error('Dimensioni non corrispondenti tra X e y target in blocco %d, elettrodo %d', b, k);
        %end
        % Stima dei coefficienti della Ridge Regression: (X'X + lambda*I) \
        % X'y
        % Matrice identità per la regolarizzazione
         
        coefs_k{b, k} = (X' * X + lambda * I) \ (X' * y_k);
        % Predizione della risposta neurale
        y_hat_k{b, k} = X * coefs_k{b, k};
      
    end
end

% SCimmia S
for b = 1:n_blocchi
     % matrice di regressori (una per blocco, costante tra elettrodi) 
    X=X_S{b}
    I = eye(size(X,2)); 
    for s = 1:n_channels_s
        % target (segnale neurale per quel blocco ed elettrodo)
        % sono le y osservate per la scimmia 
        y_s = Y_S{b, s}; 
        % check su dimensioni
        % if size(X,1) ~= size(Y_target,1)
         %   error('Dimensioni non corrispondenti tra X e y target in blocco %d, elettrodo %d', b, k);
        %end
        % Stima dei coefficienti della Ridge Regression: (X'X + lambda*I) \
        % X'y
        % Matrice identità per la regolarizzazione
      
        coefs_s{b, s} = (X' * X + lambda * I) \ (X' * y_s);
        % Predizione della risposta neurale
        y_hat_s{b, s} = X * coefs_s{b, s};    
    end
end

%% Correlation analysis
% Questo di fatto è il punto 1 nel paragrafo Model-based brain-to-brain
% coupling  MEtodi (pag e4)
% "we develop a framework for assessing five types of generalization simultaneously:
% testing encoding model generalization
% (1) across segments of the stimulus (using 10-fold cross-validation)"
% Ora, in teoria, il modello viene testato utilizzando una 10 fold
% cross-validation 
% In ogni iterazione, un sottoinsieme di blocchi di parole viene escluso 
% dal training e utilizzato come test set 
% Dopo l'addestramento Le risposte neurali predette (`y_hat`) per il test
% set vengono calcolate moltiplicando la matrice delle embedding del test 
% set con i coefficienti stimati. Si calcola la correlazione tra i valori
% predetti y_hat e le osservate.
% Il processo viene ripetuto per tutti i fold della cross-validation.
% Alla fine, si ottiene una correlazione media tra predetto e osservato per
% ogni fold,che rappresenta la performance del modello nell'encoding delle
% risposte neurali

%faccio k FOLDS CV...siccome abbiamo mantenuto la dimensione temporale
%e ogni y è lunga (t_lag*n_movimenti), la cross validation non deve spezzare
% un movimento (nel senso non deve troncare un blocco fatto di t_lag
% obs...) da cui, il k di k_fold deve essere un fattore comune di t_lag e
% del numero di movimenti (sperando ci sia e uno dei due non sia
% primo). 
% lunghezza del singolo blocco (osservazioni post movimento per lag)
t_lag=length((dati_S{1}{1}(:, 1)))
n_movements=n_trial
% lunghezza delle y
l_y=n_movements


% fattore comune più alto
ff=gcd(t_lag, n_movements)
%factors=divisors(sym(ff))

k_Fold = 4; % Numero di fold per la cross-validation
% qui, diversamente dal paper. dove, come detto abbiamo solo una
% osservazione per parola(nell'embedding X) noi ne abbiamo (per movimento
% ovviamente) un numero pari alla durata dei blocchi...cosa di cui devo
% tenere conto nella cross validation
foldIndices = ceil(linspace(1, l_y+1, k_Fold+1))

% Inizializzazione delle strutture per i coefficienti, predizioni e
% correlazioni  
% Scimmia K
coefs_k_cv = cell(n_blocchi, n_channels_k);
y_hat_k_cv = cell(n_blocchi, n_channels_k);
corrs_k_cv = zeros(n_blocchi, n_channels_k, k_Fold);
% Scimmia S
coefs_s_cv =cell(n_blocchi, n_channels_s);
y_hat_s_cv =cell(n_blocchi, n_channels_s);
corrs_s_cv =  zeros(n_blocchi, n_channels_s, k_Fold);


% Scimmia k
for b = 1:n_blocchi
    X_= X_K{b};
    I = eye(size(X_,2));
    for e = 1:n_channels_k
        % target (segnale neurale per quel blocco ed elettrodo)
        % sono le y osservate per la scimmia 
        y_ = Y_K{b, e};
        for k = 1:k_Fold
            % Definisco train/test
            testIdx = foldIndices(k):foldIndices(k+1)-1;
            trainIdx = setdiff(1:l_y, testIdx);
        
            X_train = X_(trainIdx, :);
            y_train = y_(trainIdx, :);
            X_test = X_(testIdx, :);
            y_test = y_(testIdx, :);

            % Stima dei coefficienti della Ridge Regression: (X'X + lambda*I) \
            % X'y
            % Matrice identità per la regolarizzazione
            
            % 
            coefs_k_cv{b, e, k} = (X_train' * X_train + lambda * I) \ (X_train' * y_train);

            % Predicted y
            y_hat_test_k_cv = X_test * coefs_k_cv{b, e, k};

            
            % correlazione predicted observed (speaker)
            corrs_k_cv(b, e, k) = corr(y_test, y_hat_test_k_cv,'rows','pairwise');
           
        end
    end
end

% Media della correlazione 
% alla fine mi ritrovo con una matrice di correlazione per blocco per 
% per elettrodo
corrs_mean_k_cv = mean(corrs_k_cv, 3,'omitnan');
% come loro prendo una soglia (che loro caclolano empiricamente con permutazioni
% e tengo solo le correlazioni oltre quella soglia--
r_threshold = 0.05;  
corr_k_cv_ = corrs_mean_k_cv > r_threshold;
corr_k_cv = corrs_mean_k_cv .* corr_k_cv_ ;


% Scimmia S
for b = 1:n_blocchi
    X_= X_S{b}
    I = eye(size(X_,2));

    for e = 1:n_channels_s
        % target (segnale neurale per quel blocco ed elettrodo)
        % sono le y osservate per la scimmia 
        y_ = Y_S{b, e};
        %% y osservate listener
        %y_l = Y_l{b, e};
        for k = 1:k_Fold
            % Definisco train/test
            testIdx = foldIndices(k):foldIndices(k+1)-1;
            trainIdx = setdiff(1:l_y, testIdx);
        
            X_train = X_(trainIdx, :);
            y_train = y_(trainIdx, :);
            X_test = X_(testIdx, :);
            y_test = y_(testIdx, :);

            % Stima dei coefficienti della Ridge Regression: (X'X + lambda*I) \
            % X'y
            % Matrice identità per la regolarizzazione
            
            X_test = X_(testIdx, :);
            y_test = y_(testIdx, :);

            % Stima dei coefficienti della Ridge Regression: (X'X + lambda*I) \
            % X'y
            % Matrice identità per la regolarizzazione
            % 
            coefs_s_cv{b, e, k} = (X_train' * X_train + lambda * I) \ (X_train' * y_train);

            % Predicted y
            y_hat_test_s_cv = X_test * coefs_s_cv{b, e, k};

            
            % correlazione predicted observed (speaker)
            corrs_s_cv(b, e, k) = corr(y_test, y_hat_test_s_cv,'rows','pairwise');
           
        end
    end
end
corrs_mean_s_cv = mean(corrs_s_cv, 3,'omitnan');
% come loro prendo una soglia (che loro caclolano empiricamente con permutazioni
% e tengo solo le correlazioni oltre quella soglia--
r_threshold = 0.05;  
corr_s_cv_ = corrs_mean_s_cv > r_threshold;
corr_s_cv = corrs_mean_s_cv .* corr_s_cv_ ;
% Display risultati
disp("Correlazione media per ogni canale e blocco - withih subject- K:");
disp(corr_k_cv);
disp("Correlazione media per ogni canale e blocco - withih subject- S:");
disp(corr_s_cv);


%%%%%%%%%%%%%%%%%%%%%%% 4) Brain  to Brain coupling %%%%%%%%%%%%%%%%%

% Per questa parte mi occorrono gli stessi neuroni, o meglio, un numero di
% neuroni (canali) uguale per scimmia...dal momento che K ha 202 neuroni e
% S ne ha 184, seleziono 184 nuroni per K. Lo faccio sulla base degli shap
% values (in python) con il quale ho creato un "ranking" di importana di
% sulla base del quale ne tolgo 18

%% Neuroni da rimuovere
to_remove=[97,156,109,165,25,58,153,4,190,10,33,65,3,135,103,48,118,186]
Y_K_reduced=Y_K
Y_K_reduced(:,to_remove)=[]
coefs_k_cv_reduced=coefs_k_cv
coefs_k_cv_reduced(:,to_remove,:)=[]
%%%%%%%%%%%%%%%%% punto 2 nel paragrafo Model-based brain-to-brain coupling MEtodi (pag e4)
%%%%%%%%%%%%%%%%%%%%  GENERALIZATION ACROSS SUBJECTS ######################
% "we develop a framework for assessing five types of generalization simultaneously:
% testing encoding model generalization
% (2) across subjects (within speaker–listener dyads)"
% considero i coefficienti stimati  dello speaker per capire se l'attività
% neurale dello speaker è predittiva per  il listener e viceversa e se la
% relazione tra le due attività cerebrali segue schemi coerenti 

% di fatto le y previste non cambiano da quelle stimate nel modello within-
% subject...si usano le stesse X (embedding) e gli stessi coefficienti

k_Fold = 4; % Numero di fold per la cross-validation
% qui, diversamente dal paper. dove, come detto abbiamo solo una
% osservazione per parola(nell'embedding X) noi ne abbiamo (per movimento
% ovviamente) un numero pari alla durata dei blocchi...cosa di cui devo
% tenere conto nella cross validation
foldIndices = ceil(linspace(1, l_y+1, k_Fold+1))
%Inizializzazione della struttura per la correlazione inter-soggetto
% i canali a questo punto sono pari allla dimensione della struttura Y_S

n_channels=length(Y_S)

% da S a K
corrs_inter_K_S = zeros(n_blocchi, n_channels, k_Fold);
% da K a S
corrs_inter_S_K = zeros(n_blocchi, n_channels, k_Fold);

for b = 1:n_blocchi
    %% X varia per i soggetti (ognuno ha la sua manifold) e per blocco/lag
    X_k=X_K{b}
    X_s=X_S{b}
    I = eye(size(X_s,2));
     for e = 1:n_channels
        y_k = Y_K_reduced{b, e}; % Dati neurali K
        y_s = Y_S{b, e}; % Dati neurali  S

        for k = 1:k_Fold
            % Indici del test (come sopra)
            testIdx = foldIndices(k):foldIndices(k+1)-1;
            % Dati test del listener
            X_test_k = X_k(testIdx, :);
            X_test_s = X_s(testIdx, :);
            y_test_k = y_k(testIdx, :);
            y_test_s = y_s(testIdx, :);
            

            % Uso i coefficienti di K su S e viceversa
            % senza ristimarli!

            y_hat_k = X_test_k*coefs_s_cv{b, e, k};
            y_hat_s = X_test_s*coefs_k_cv_reduced{b, e, k};

            % Correlazione tra osservato (listener) e predetto (da speaker)
            corrs_inter_K_S(b, e, k) = corr(y_test_k, y_hat_k);
            corrs_inter_S_K(b, e, k) = corr(y_test_s, y_hat_s);

        end
    end
end

% di fatto mi trovo con delle  matrici 3d con le correlazioni per elettrodi 
% per lags per folds [n_blocchi, n_channels, n_folds]

%%%%%% COSTRUZIONE HEATMAPS lag lag fig 3a %%%%%%%%%%
n_lags=n_blocchi; n_folds=k_Fold; 

% Calcolo la media across electrodes per ogni fold (come fanno in Python)
%  ottengo matrici (lags, folds)
corrs_K_S_avg = squeeze(mean(corrs_inter_K_S, 2,"omitnan")); 
corrs_S_K_avg = squeeze(mean(corrs_inter_S_K, 2,"omitnan"));

% Offset per evitare problemi di numeri molto piccoli
epsilon = 1e-4; 

% Aggiungiamo epsilon ai valori troppo vicini a zero
corrs_K_S_avg(abs(corrs_K_S_avg) < epsilon) = sign(corrs_K_S_avg(abs(corrs_K_S_avg) < epsilon)) * epsilon;
corrs_S_K_avg(abs(corrs_S_K_avg) < epsilon) = sign(corrs_S_K_avg(abs(corrs_S_K_avg) < epsilon)) * epsilon;

% Inizializzo la matrice lag-lag
corr_matrix = zeros(n_lags, n_lags);

% Calcolo  l correlazione per ogni combinazione di lag K-S
for i = 1:n_lags
    for j = 1:n_lags
        % Vettore di 3 valori per il lag i (K → S)
        K_vector = corrs_K_S_avg(i, :); % (1 × 4)
        
        % Vettore di 3 valori per il lag j (S → K)
        S_vector = corrs_S_K_avg(j, :); % (1 × 4)
        
        % Calcolo la correlazione tra i due vettori
        corr_matrix(i, j) = corr(K_vector', S_vector');
    end
end


% ora medio la matrice di correlazione across folds
%corr_matrix = mean(corr_matrix_folds, 3);

%% per il plot
%% da migliorare e magari mettere insieme alcuni lag
% time range (vado a spanne...da perfezionare...sono circa 400 ms dopo il movimento)
 % Array dei tempi associati ai lag
time_range = linspace(0, 400, n_lags);
 % Seleziona 5 tick equidistanti
tick_indices = round(linspace(1, n_lags, 5));
tick_labels = arrayfun(@(x) sprintf('%.1fms', time_range(x)), tick_indices, 'UniformOutput', false);

% Visualizzazione della heatmap
figure;
imagesc(corr_matrix);
colormap('hot');
colorbar;
axis square;
xlabel('Lag S (s)');
ylabel('Lag K (s)');
title('Heatmap della correlazione lag-lag tra S e K');
set(gca, 'XTick', tick_indices, 'XTickLabel', tick_labels, ...
         'YTick', tick_indices, 'YTickLabel', tick_labels);

%%%%%%%%%%%%%%%%% punti 3-4-5 nel paragrafo Model-based brain-to-brain coupling
% MEtodi (pag e4) we develop a framework for assessing five types of generalization simultaneously:
% testing encoding model generalization
% 3)across different brain regions (e.g., from SM to STG electrodes), 
% 4) across tasks/processes (speaking/production and listening/comprehension)
% 5) across lags (e.g., speaker pre-word onset to listener post-word onset).

%% sostanzialmente fa la stessa cosa (calcolo correlazioni, media delle stesse etc)
% cambiando però i riferiemnti; nel punto 3 fa la stessa cosa ma usando
% aree diverse del cervello (stima quindi il modello considerando alcuni
% elettrodi e vede la capacità predittiva su altri elettrodi


