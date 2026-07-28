@echo off
REM === Crea la struttura del framework NeuroBridge ===

REM cartella principale
mkdir src

@echo off
REM === Struttura NeuroBridge (src-based) ===

REM cartelle
mkdir src
mkdir src\neurobridge
mkdir src\neurobridge\data
mkdir src\neurobridge\sampling
mkdir src\neurobridge\encoders
mkdir src\neurobridge\losses
mkdir src\neurobridge\train
mkdir configs
mkdir experiments
mkdir tests
mkdir src\neurobridge\output\projects\hasson_common_space\


REM __init__.py per i package
echo # init per il pacchetto neurobridge > src\neurobridge\__init__.py
echo # init per il modulo data > src\neurobridge\data\__init__.py
echo # init per il modulo sampling > src\neurobridge\sampling\__init__.py
echo # init per il modulo encoders > src\neurobridge\encoders\__init__.py
echo # init per il modulo losses > src\neurobridge\losses\__init__.py
echo # init per il modulo train > src\neurobridge\train\__init__.py

REM file placeholder (con commento descrittivo)
echo # dataset e gestione finestre EEG > src\neurobridge\data\dataset.py
echo # riuso vecchie fuznioni > src\neurobridge\utils\io.py
echo #  > src\neurobridge\utils\resample.py
echo # d > src\neurobridge\viz\plots.py

echo # funzioni di sampling contrastivo (positivi/negativi) > src\neurobridge\sampling\labelled.py
echo # encoder temporale (es. CNN/Transformer) > src\neurobridge\encoders\temporal_cnn.py
echo # definizione della loss contrastiva (es. InfoNCE) > src\neurobridge\losses\infonce.py
echo # training loop e funzioni di addestramento > src\neurobridge\train\loop.py
echo # entry point principale per testare il framework > main.py

echo [OK] Struttura NeuroBridge creata/aggiornata con placeholder!
pause



  REM --- Se non c'è già __init__.py, crealo ---
  if not exist %%d\__init__.py (
    echo. > %%d\__init__.py
  )
)
