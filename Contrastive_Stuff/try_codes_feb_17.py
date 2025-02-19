# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 16:39:24 2025

@author: loren
"""

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpld3
fig = plt.figure(figsize=(8, 6))
n_traj=8
trial_length
output_folder = out_dir
#os.makedirs(output_folder, exist_ok=True)
#ax = Axes3D(fig)
ax =fig.add_subplot(111,projection = '3d')
ax.set_title('direction-averaged embedding', fontsize=10, y=-0.1)
# Creazione della sfera unitaria
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, color='lightblue', alpha=0.05, edgecolor='k',linewidth=0.2)
for i in range(n_traj):
        direction_trial = (l_dir == i)
        trial_avg = z_[direction_trial, :] .reshape(-1, trial_length-ww,
                                                              3).mean(axis=0)
        trial_avg_normed = trial_avg/np.linalg.norm(trial_avg, axis=1)[:,None]
        #print(trial_avg.shape, direction_trial.shape, trial_avg_normed.shape)
        #trial_avg_normed = trial_avg/np.linalg.norm(trial_avg, axis=1)[:,None]
        print(trial_avg.shape, direction_trial.shape, trial_avg_normed.shape)

        ##ax.plot(trial_avg_normed[:, 0],
        #           trial_avg_normed[:, 1],
         #          trial_avg_normed[:, 2],
        #           marker='o',
                   #c='r')
        ax.plot(trial_avg_normed[:, 0],
                   trial_avg_normed[:, 1],
                   trial_avg_normed[:, 2],
                
                    color=plt.cm.hsv( i/n_traj),
                    linewidth=2, alpha=0.6, label=f"Dir {i}")
        ax.scatter(trial_avg_normed[0, 0], 
           trial_avg_normed[0, 1], 
           trial_avg_normed[0, 2], 
           color='red', marker='o', s=20, label="Start" if i == 0 else "")

        ax.scatter(trial_avg_normed[-1, 0], 
           trial_avg_normed[-1, 1], 
           trial_avg_normed[-1, 2], 
           color='blue', marker='x', s=20, label="End" if i == 0 else "")
# Aggiunta della griglia e legenda
ax.grid(True, linestyle="--", linewidth=0.5)
ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=8)
plt.show()

html_fig = mpld3.fig_to_html(fig)
# Save the HTML representation to a file
with open("interactive_plot.html", "w") as f:
    f.write(html_fig)

output_file = os.path.join(output_folder, "plot_interattivo.html") 
mpld3.save_html(fig, output_file)
ax.axis('on')



trial_avg_normed = trial_avg/np.linalg.norm(trial_avg, axis=1)[:,None]