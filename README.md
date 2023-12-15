15.12.2023: Carico file main_rat, hip_models e Fig2_mod

N.B.! Nel main si lanciano gli altri due script che generano output e figure. 
      Sul main va cambiata la main_path che è a directory dove vogliamo mettere tutto
      e nella quale verrà contestualmente creata una cartella "images" in cui compaiono 
      tutte le figure (carico anche quella cartella in cui ho messo già le figure generate
      con 4000, 5000 e 10000 iterazioni). Contestualmente si generano anche gli output in formato
      .mat per fare grafici ed elaborazioni su Matlab. 

      main path è anche argomento (input) dei due script (quindi va cambiata SOLO sul main). 
      Se vogliamo generare gli output a livello intermedio, basta andare nella parte del main

      if __name__=="__main__":
        main() 
     
     e cambiarla in 

     if __name__=="__main__":
      dd, err_loss, mod_pred= main()




