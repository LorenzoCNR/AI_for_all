15.12.2023: Carico file main_rat, hip_models e Fig2_mod

N.B.! Nel main si lanciano gli altri due script che generano output e figure. 
      Sul main va cambiata la main_path che è a directory dove vogliamo mettere tutto
      e nella quale verrà contestualmente creata una cartella "images" in cui compaiono 
      tutte le figure (carico anche quella cartella in cui ho messo già le figure generate
      con 4000, 5000 e 10000 iterazioni). Contestualmente si generano anche gli output in formato
      .mat per fare grafici ed elaborazioni su Matlab.
      Inoltre, la cartella figure ottenute contiene vecchie figure (ci sono anche quelle fatte sui dati EEG di Mirco in prima battuta). 
      La cartella MatLab_files contiene tutti gli script per generare i grafici più il principale, cebra_script che fa partire tutto su MatLab (in Lavorazione).
      La cartella Third_Party contien script per altri algoritmi (nello specifico pi_vae che va raffinato). 

      main path è anche argomento (input) dei due script (quindi va cambiata SOLO sul main). 
      Se vogliamo generare gli output a livello intermedio, basta andare nella parte del main

      if __name__=="__main__":
        main() 
     
     e cambiarla in 

     if __name__=="__main__":
      dd, err_loss, mod_pred= main()


19.12.2023: Carico file wrap_ML e wrap_py per output per elaborazioni direttamente da MatLab




