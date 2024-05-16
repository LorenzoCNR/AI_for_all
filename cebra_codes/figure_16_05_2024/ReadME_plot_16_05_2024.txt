La struttura è la seguente:

file models_1.py
file model_params_1.yaml
file plots_1.py


il file models_1 integra diverse funzioni:
- caricamento dei dati (load dataset for rat)
- caricamento dei parametri (load params) che ha come input il file model_params_1.yaml
	i parametri sono divisi in fissi e 'grid' quelli in grid li giro
- creazione del modello (create model) con input il modello scelto (cebra nelle sue declinazioni, Umap,     
	Tsne) e i parameteri caricati dallo yaml 
- processing dei dati secondo il modello (run model e transform)...supervised nel caso di hybrid e behavior (cebra)
- salvataggio dei risultati con creazione del dataset e del gruppo se non esistono e possibilità di 
	sostituire la manifiold se esistente che avrà per nome la data e i valori dei parametri in grid 

il main vuole in input:
- directory di input
- directory di output
- nome del ratto
- nome del file output (percorso e nome del file hd5)
- nome del file parametri.yaml (percorso e nome )
- tipo di modello (cebra_time, cebra_behavior, cebra_hybrid, umap, tsne)....
- se usare o meno la griglia di parametri (nel caso negativo, di quelli in griglia usa il primo della lista)
- replace (True o False) se True sostituisce all'interno del gruppo il dataset


il file plots ....