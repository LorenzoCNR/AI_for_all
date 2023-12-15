#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 14:15:23 2023

@author: zlollo
"""

#!pip install -r requirements.txt
#### Visualiza struttura  e contenuto directory
import sys
import os
from pathlib import Path  # Aggiungi questa linea+
##os.chdir(r'/home/zlollo/CNR/git_out_cebra/cebra-figures-main')
    
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pprint
import matplotlib.colors as mcolors
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import typing
from matplotlib.markers import MarkerStyle
from matplotlib.collections import LineCollection
import scipy.stats
from statsmodels.sandbox.stats.multicomp import get_tukey_pvalue
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.oneway import anova_oneway
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
    
def Fig2_rat_hip(data, err_loss, mod_pred, base_path):


    os.chdir(base_path)
    IMAGES_PATH = Path() / "images" 
    IMAGES_PATH.mkdir(parents=True, exist_ok=True)
    
    # def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    #     path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    #     #fig = plt.gcf() 
    #     if tight_layout:
    #         plt.tight_layout()
    #     plt.savefig(path, format=fig_extension,bbox_inches='tight')
    
    def save_fig(fig, fig_id, tight_layout=False, fig_extension="png"):
        path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
        if tight_layout:
            fig.tight_layout()
        fig.savefig(path, format=fig_extension, bbox_inches='tight')
        

    # def visualizza_struttura(directory):
    #     for dirpath, dirnames, filenames in os.walk(directory):
            
    #         # directory corrente
    #         print(f"Directory: {dirpath}")
    
    #         # sottodirectory
    #         for dirname in dirnames:
    #             print(f"  - Subdirectory: {os.path.join(dirpath, dirname)}")
    
    #         # Stampa i file
    #         for filename in filenames:
    #             print(f"  - File: {os.path.join(dirpath, filename)}")
    ## ex
    # directory_da_esaminare = '/home/zlollo/CNR/Cebra_for_all/third_party/pivae'


    ################################# FIG 2B ##à######################################
    ####cebra con embedding derivate da ipotesi (posizione), shuffle,
    #### time  e tempo+comportamento 
    ### fucsia verso celeste sono i cm in direzione six 0 celeste, fucsia 160
    ### giallo verso nero sono i cm in direz dex 0 nerop giallo 160
    
    method_viz = data["visualization"]
    
    ### visualiziamo gli spazi embedded generati con CEBRA
    
    
    # fig = plt.figure(figsize=(30, 10))
    # for i, model in enumerate(["hypothesis", "shuffled", "discovery", "hybrid"]):
    #     ax = fig.add_subplot(1, 4, i + 1, projection="3d")
    #     emb = method_viz[model]["embedding"]
    #     label = method_viz[model]["label"]
    #     r = label[:, 1] == 1
    #     l = label[:, 2] == 1
    #     idx1, idx2, idx3 = (0, 1, 2)
    #     if i == 3:
    #         idx1, idx2, idx3 = (1, 2, 0)
    #     ax.scatter(
    #         emb[l, idx1], emb[l, idx2], emb[l, idx3], c=label[l, 0], cmap="cool",
    #         s=0.1
    #     )
    #     ax.scatter(emb[r, idx1], emb[r, idx2], emb[r, idx3], c=label[r, 0], s=0.1)
    #     ax.axis("off")
    #     ax.set_title(f"{model}", fontsize=20)
    
    
    ### shuffle rompe la correlazione tra attività neurale e di comportamento
    
    # Preparazione della figura
    fig = plt.figure(figsize=(20, 5))
    
    
    # Variabili per gestire la colorbar unica
    cmap1 = plt.get_cmap('cool')
    cmap2 = plt.get_cmap('summer')
    label = data['visualization']['Hypothesis: position']["label"]
    norm = plt.Normalize(vmin=label[:, 0].min(), vmax=label[:, 0].max())

 #    norm = plt.Normalize(vmin=label[:, 0].min(), vmax=label[:, 0].max())
    
     # Ciclo per ogni modello
    for i, model in enumerate(['Hypothesis: position', 'Shuffled Labels', 
          'Discovery: time only','Hybrid: time + behavior']):
          # Preparazione del grafico
          fig = plt.figure(figsize=(10, 5))
          ax = fig.add_subplot(111, projection="3d")
        
          emb = method_viz[model]["embedding"]
          label = method_viz[model]["label"]
          idx1, idx2, idx3 = (0, 1, 2)
          if i == 3:
              idx1, idx2, idx3 = (1, 2, 0)
        
          r = label[:, 1] == 1
          l = label[:, 2] == 1
        
          # Plot per sinistra e destra
          ax.scatter(emb[l, idx1], emb[l, idx2], emb[l, idx3], c=label[l, 0], 
                    cmap=cmap1, norm=norm, s=0.1, label='Left')
          ax.scatter(emb[r, idx1], emb[r, idx2], emb[r, idx3], c=label[r, 0], 
                     cmap=cmap2, norm=norm, s=0.1, label='Right')
        
          # Axes Removal
          ax.axis("off")
        
          # Titolo
          ax.set_title(f"{model}", fontsize=20)
 
      # Aggiunta delle annotazioni di testo per indicare le direzioni
          #ax.text2D(0.05, 0.95, "Left", transform=ax.transAxes)
          #ax.text2D(0.95, 0.95, "Right", transform=ax.transAxes)
    
   # Aggiungi una colorbar per ciascun grafico
          sm = plt.cm.ScalarMappable(cmap=cmap1, norm=norm)
          sm.set_array([])
          # # Colorbar 1
          cbar = plt.colorbar(sm, ax=fig.axes, orientation='vertical', fraction=0.01, pad=0.15)
          cbar.set_label('left')
        
          sm2 = plt.cm.ScalarMappable(cmap=cmap2, norm=norm)
          sm2.set_array([])
          cbar2 = plt.colorbar(sm2, ax=fig.axes, orientation='vertical', fraction=0.01, pad=0.15)  # Imposta pad a un valore diverso
          cbar2.set_label('right')




          save_fig(fig,f"plot_{model}", tight_layout=False, fig_extension="eps")        
          # Mostra il grafico
          plt.show()

   #  # Variabili per gestire la colorbar unica
   #  cmap1 = plt.get_cmap('cool')
   #  cmap2 = plt.get_cmap('summer')
   #  label = data['visualization']["hypothesis"]["label"]

    
     # Ciclo per ogni modello
    # for i, model in enumerate(["hypothesis", "shuffled", "discovery", "hybrid"]):
    #       ax = fig.add_subplot(1, 4, i + 1, projection="3d")
    #       emb = method_viz[model]["embedding"]
    #       label = method_viz[model]["label"]
    #       # Indici per selezionare i dati corretti
    #       idx1, idx2, idx3 = (0, 1, 2)
    #       if i == 3:
    #           idx1, idx2, idx3 = (1, 2, 0)
    
    #       # Maschere per destra e sinistra
    #       r = label[:, 1] == 1
    #       l = label[:, 2] == 1
    
    #       # Plot per sinistra e destra
    #       ax.scatter(emb[l, idx1], emb[l, idx2], emb[l, idx3], c=label[l, 0], 
    #                 cmap=cmap1, norm=norm, s=0.1)
    #       ax.scatter(emb[r, idx1], emb[r, idx2], emb[r, idx3], c=label[r, 0], 
    #                 cmap=cmap2, norm=norm, s=0.1)
        
    #       # Axes Removal
    #       ax.axis("off")
    
    #       # 
    #       ax.set_title(f"{model}", fontsize=20)
    
    #       # Aggiunta delle annotazioni di testo per indicare le direzioni
    #       #ax.text2D(0.05, 0.95, "Left", transform=ax.transAxes)
    #       #ax.text2D(0.95, 0.95, "Right", transform=ax.transAxes)
    
    # sm = plt.cm.ScalarMappable(cmap=cmap1, norm=norm)
    # sm.set_array([])
    # # Colorbar 1
    # cbar = plt.colorbar(sm, ax=fig.axes, orientation='vertical', fraction=0.01, pad=0.02)
    # cbar.set_label('left')
    
    #   # Colorbar 2
    # sm2 = plt.cm.ScalarMappable(cmap=cmap2, norm=norm)
    # sm2.set_array([])
    # cbar2 = plt.colorbar(sm2, ax=fig.axes, orientation='vertical', fraction=0.01, pad=0.05)  # Imposta pad a un valore diverso
    # cbar2.set_label('right')
    # save_fig("2b", tight_layout=False, fig_extension="eps")
    # plt.show()
    
    
    ################################# FIG 2C ##à######################################
    ### Embeddings with position-only, direction-only and position+direction,
    ### Shuffled position-only, direction-only,p+d shuffled for hypothesis testing. 
    ### The loss function can be used as a metric for embedding quality.
    
    ### occhio che qui ho cambiato le dimensioni
        
    hypothesis_viz = data["hypothesis_testing"]["viz"]
    
    fig = plt.figure(figsize=(15, 10))
    titles = {
        "pos": "Position only",
        "dir": "Direction only",
        "posdir": "Position+Direction",
        "pos-shuffled": "Position, shuffled",
        "dir-shuffled": "Direction, shuffled",
        "posdir-shuffled": "P+D, shuffled",
    }
    for i, model in enumerate(
        ["pos", "dir", "posdir", "pos-shuffled", "dir-shuffled", "posdir-shuffled"]
    ):
        emb = hypothesis_viz[model]
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        idx1, idx2, idx3 = (0, 1, 2)
        ax.scatter(emb[:, idx1], emb[:, idx2], emb[:, idx3], c="gray", s=0.1)
        ax.axis("off")
        ax.set_title(f"{titles[model]}", fontsize=20)
    save_fig(fig,"2c", tight_layout=False, fig_extension="eps")
    plt.show

    
    ########### perdita legata ai modelli
  
    hypothesis_loss = data["hypothesis_testing"]["loss"]
    fig = plt.figure(figsize=(4, 4))
    ax = plt.subplot(111)
    titles = {
        "pos": "Position only",
        "dir": "Direction only",
        "posdir": "Position+Direction",
        "pos-shuffled": "Position, shuffled",
        "dir-shuffled": "Direction, shuffled",
        "posdir-shuffled": "P+D, shuffled",
    }
    alphas = {
        "pos": 0.3,
        "dir": 0.6,
        "posdir": 1,
        "pos-shuffled": 0.3,
        "dir-shuffled": 0.6,
        "posdir-shuffled": 1,
    }
    for model in [
        "pos",
        "dir",
        "posdir",
        "pos-shuffled",
        "dir-shuffled",
        "posdir-shuffled",
    ]:
        if "shuffled" in model:
            c = "gray"
        else:
            c = "deepskyblue"
        ax.plot(hypothesis_loss[model], c=c, alpha=alphas[model], label=titles[model])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Iterations", fontsize=15)
    ax.set_ylabel("InfoNCE Loss", fontsize=15)
    ax.set_title('Hypothesis Testing, Training Loss', fontsize=20)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(bbox_to_anchor=(1, 0.5), frameon=False, fontsize=10)
    save_fig(fig,"2c_Loss", tight_layout=False, fig_extension="eps")

    
    ################################# FIG 2D #######################################
    # We utilized the hypothesis-driven (position) or the shuffle (erroneous)
    #to decode the position of the rat, which produces a large difference 
    #in decoding performance: position+direction is 73.35% vs. -49.90% shuffled
    # and median absolute error 5.8 cm vs 44.7 cm. Purple line is decoding
    # from the hypothesis-based latent space, dashed line is shuffled.
    #data=dd
    
    # select timesteps
    start_idx = 320
    length = 700
    history_len = 700
    
    # load data
    #fs = data_fig_2d["embedding_all"].item()
    fs= data["hypothesis_testing"]["viz"]['posdir']
    #data_fig_2d["embedding_all"].item().shape
    #test_fs = data_fig_2d["embedding_test"].item()
    test_fs = mod_pred['cebra_posdir_test']
    #labels = data_fig_2d["true_all"].item()
    labels = label
    
    test_labels = mod_pred['label_test']
    #pred = data_fig_2d["prediction"].item()
    pred=mod_pred['pred_posdir_decode']
    #pred_shuffle = data_fig_2d["prediction_shuffled"].item()
    pred_shuffle=mod_pred['pred_posdir_shuffled_decode']
    # plot
    fig = plt.figure(figsize=(4, 2), dpi=300)
    linewidth = 2
    ax1_traj = plt.gca()
    
    framerate = 25 / 1000
    
    true_trajectory = ax1_traj.plot(
        framerate * np.arange(length, step=2),
        test_labels[start_idx : start_idx + length, 0][np.arange(length, step=2)] * 100,
        "-",
        c="k",
        label="Ground Truth",
        linewidth=linewidth,
    )
    
    (pred_trajectory,) = ax1_traj.plot(
        framerate * np.arange(length, step=2),
        pred[start_idx : start_idx + length, 0][np.arange(length, step=2)] * 100,
        c="#6235E0",
        label="CEBRA-Behavior",
        linewidth=linewidth,
    )
    
    (pred_shuffle_trajectory,) = ax1_traj.plot(
        framerate * (np.arange(length, step=10) + 2),
        pred_shuffle[start_idx : start_idx + length, 0][np.arange(length, step=10)] * 100,
        "--",
        c="gray",
        label="CEBRA-Shuffle",
        linewidth=linewidth,
    )
    
    ax1_traj.set_yticks(np.linspace(0, 160, 5))
    
    ax1_traj.spines["right"].set_visible(False)
    ax1_traj.spines["top"].set_visible(False)
    
    legend = ax1_traj.legend(
        loc=(0.6, 1.0),
        frameon=False,
        handlelength=1.5,
        labelspacing=0.25,
        fontsize="x-small",
    )
    
    ax1_traj.set_xlabel("Time [s]")
    ax1_traj.set_ylabel("Position [cm]")
    
    ax1_traj.set_xlim([-1, 17.5])
    ax1_traj.set_xticks(np.linspace(0, 17.5, 8))
    
    ax1_traj.spines["bottom"].set_bounds(0, 17.5)
    ax1_traj.spines["left"].set_bounds(0, 160)
    save_fig(fig,"2d_line", tight_layout=False, fig_extension="eps")
    #save_fig("figure_2d_lines.svg", bbox_inches="tight", transparent=True)

    plt.show()



#### visualizziamo i risultati del decoding

    
    # Sample data (replace with your data)
    first_six_keys = list(err_loss.keys())[:6]
    first_six_values = [err_loss[key] for key in first_six_keys]
    
    fig = plt.figure(figsize=(10, 4))
    
    ax1 = plt.subplot(121)
    
    width = 0.5
    fig = plt.figure(figsize=(10, 4))
    
    ax1 = plt.subplot(121)
    
    width = 0.5
    x = np.arange(len(first_six_keys))
    
    ax1.bar(x, first_six_values, width, color='gray')
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(first_six_keys)
    ax1.set_ylabel('Value')
    
    plt.xticks(rotation=45) # Ruota le etichette dell'asse x per una migliore leggibilità
    
    
    
    ax2 = plt.subplot(122)
    ax2.scatter(err_loss['loss_posdir_decode'],err_loss['error_posdir_decode'], s=50, c='red', label = 'position+direction')
    ax2.scatter(err_loss['loss_pos_decode'],err_loss['error_pos_decode'], s=50, c='green', alpha = 0.3, label = 'position_only')
    ax2.scatter(err_loss['loss_dir_decode'],err_loss['error_dir_decode'], s=50, c='deepskyblue', alpha=0.6,label = 'direction_only')
    ax2.scatter(err_loss['loss_posdir_decode_shuffled'],err_loss['error_posdir_decode_shuffled'], s=50, c='gray', label = 'pos+dir, shuffled')
    ax2.scatter(err_loss['loss_pos_decode_shuffled'],err_loss['error_pos_decode_shuffled'], s=50, c='black', alpha = 0.3, label = 'position, shuffled')
    ax2.scatter(err_loss['loss_dir_decode_shuffled'],err_loss['error_dir_decode_shuffled'], s=50, c='brown', alpha=0.6,label = 'direction, shuffled')
    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xlabel('InfoNCE Loss')
    ax2.set_ylabel('Decoding Median Err.')
    plt.legend(bbox_to_anchor=(1,1), frameon = False )
    save_fig(fig,"2d_Bar_Scatter", tight_layout=False, fig_extension="eps")
    
    plt.show()



### da aggiornare####
#if __name__ == "__main__":
#    data=....5
#    cebra_posdir_test=-...

#    Fig2_rat_hip(data, cebra_posdir_test, label_test, pred_posdir_decode, pred_posdir_shuffled_decode):



    
    ######################################################################################
    ################################### FINO A QUI #######################################
    ######################################################################################
    #### performance across models
'''
    os.chdir(r'/home/zlollo/CNR/git_out_cebra/cebra-figures-main')
    
    ROOT = pathlib.Path(r'/home/zlollo/CNR/git_out_cebra/cebra-figures-main/data')
    
    def recover_python_datatypes(element):
        if isinstance(element, str):
            if element.startswith("[") and element.endswith("]"):
                if "," in element:
                    element = np.fromstring(element[1:-1], dtype=float, sep=",")
                else:
                    element = np.fromstring(element[1:-1], dtype=float, sep=" ")
        return element
    
    
    def load_results(result_name):
        """Load a result file.
    
        The first line in the result files specify the index columns,
        the following lines are a CSV formatted file containing the
        numerical results.
        """
        results = {}
        for result_csv in (ROOT / result_name).glob("*.csv"):
            with open(result_csv) as fh:
                index_names = fh.readline().strip().split(",")
                df = pd.read_csv(fh).set_index(index_names)
                df = df.applymap(recover_python_datatypes)
                results[result_csv.stem] = df
        return results
    
    
    def show_boxplot(df, metric, ax, labels=None, color="C1"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sns.boxplot(
                data=df,
                y="method",
                x=metric,
                orient="h",
                order=labels,
                width=0.5,
                color="k",
                linewidth=2,
                flierprops=dict(alpha=0.5, markersize=0, marker=".", linewidth=0),
                medianprops=dict(
                    c=color, markersize=0, marker=".", linewidth=2, solid_capstyle="round"
                ),
                whiskerprops=dict(solid_capstyle="butt", linewidth=0),
                showbox=False,
                showcaps=False,
                ax=ax,
            )
            marker_style = MarkerStyle("o", "none")
            sns.stripplot(
                data=df,
                y="method",
                x=metric,
                orient="h",
                size=4,
                color="k",
                order=labels,
                marker=marker_style,
                linewidth=1,
                ax=ax,
                alpha=0.75,
                jitter=0.15,
                zorder=-1,
            )
            ax.set_ylabel("")
            sns.despine(left=True, bottom=False, ax=ax)
            ax.tick_params(
                axis="x", which="both", bottom=True, top=False, length=5, labelbottom=True
            )
            return ax
    
    
    def _add_value(df, **kwargs):
        for key, value in kwargs.items():
            df[key] = value
        return df
    
    
    def join(results):
        return pd.concat([_add_value(df, method=key) for key, df in results.items()])
    
    
    def get_metrics(results):
        for key, results_ in results.items():
            df = results_.copy()
            df["method"] = key
            df["test_position_error"] *= 100
            df = df[df.animal == 0].pivot_table(
                "test_position_error", index=("method", "seed"), aggfunc="mean"
            )
            yield df
         
    csv_path_lfads=ROOT / 'autolfads_decoding_2d_full.csv'
         
    autolfads = pd.read_csv(csv_path_lfads, index_col=0)
    
    autolfads = autolfads.rename(columns={"split": "repeat", "rat": "animal"})
    autolfads["animal"] = autolfads["animal"].apply(lambda v: "abcg".index(v[0]))
    
    results = {}  # Make sure this dictionary is initialized before you use it
    results = load_results(result_name="results_v1")
    
    
    csv_path_pivae = ROOT / 'Figure2' / 'figure2_pivae_mcmc.csv'
    results["pivae-mc"] = pd.read_csv(csv_path_pivae, index_col=0)
    results["autolfads"] = autolfads
    
    df = pd.concat(get_metrics(results)).reset_index()
    
    plt.figure(figsize=(2, 2), dpi=200)
    ax = plt.gca()
    show_boxplot(
        df=df,
        metric="test_position_error",
        ax=ax,
        color="C1",
        labels=[
            "cebra-b",
            "pivae-mc",
            "pivae-wo",
            "cebra-t",
            "autolfads",
            "tsne",
            "umap",
            "pca",
        ],
    )
    ticks = [0, 10, 20, 30, 40]
    ax.set_xlim(min(ticks), max(ticks))
    ax.set_xticks(ticks)
    ax.set_xlabel("Error [cm]")
    ax.set_yticklabels(
        [
            "CEBRA-Behavior",
            "conv-pi-VAE (MC)",
            "conv-piVAE (kNN)",
            "CEBRA-Time",
            "autoLFADS",
            "t-SNE",
            "UMAP",
            "PCA",
        ]
    )
    # plt.savefig("figure2d.svg", bbox_inches = "tight", transparent = True)
    plt.show()
    
    
    
    ############## figure 2f 
    #Visualization of the neural embeddings computed with different 
    #input dimensions
    
    topology_viz = data["topology"]["viz"]
    
    fig = plt.figure(figsize=(15, 5))
    for i, dim in enumerate([3, 8, 16]):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        emb = topology_viz[dim]
        label = topology_viz["label"]
        r = label[:, 1] == 1
        l = label[:, 2] == 1
        idx1, idx2, idx3 = (0, 1, 2)
        if i == 1:
            idx1, idx2, idx3 = (5, 6, 7)
        ax.scatter(
            emb[l, idx1], emb[l, idx2], emb[l, idx3], c=label[l, 0], cmap="cool", s=0.1
        )
        ax.scatter(
            emb[r, idx1], emb[r, idx2], emb[r, idx3], cmap="viridis", c=label[r, 0], s=0.1
        )
        ax.axis("off")
        ax.set_title(f"Dimension {dim}", fontsize=20)
        '''
        