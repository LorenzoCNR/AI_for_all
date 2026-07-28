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

il main (che può girare anche con la semplice chiamata python models_1.y, senza argomenti) vuole in input:
- directory di input
- directory di output
- nome del ratto
- nome del file output (percorso e nome del file hd5)
- nome del file parametri.yaml (percorso e nome )
- tipo di modello (cebra_time, cebra_behavior, cebra_hybrid, umap, tsne, conv_pivae (okkio che ha problemi con tf)....
- se usare o meno la griglia di parametri (nel caso negativo, di quelli in griglia usa il primo della lista)
- replace (True o False) se True sostituisce all'interno del gruppo il dataset
Ci sono già dei valori di default per ogni parametro

il file plots_1...(N.B. questo funziona da spyder e va automatizzato...)
- carica i dati del ratto per le labels
- legge il contenuto del file hdf creato con models che contiene tutti gli embedding per tutti  i framework (stampa i nomi ) finc get dataset names
-  ci sono delle funzioni (3, rename dataset, generate new name e rename all dataset in group) che servono a rendere più leggibili i nomi nei grafici (soprattutto in fase plot)
- funzione per estrarre gli embedding per framework (extract datasets)
- funzione per plottare i dati in loop (plot datasets in group) in finestre di dimensione scelta (group size = 4)
- Salvataggio figure 
