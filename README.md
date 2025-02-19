#### CEBRA CODES


### Constrastive Stuff


### Toy Examples


############################## DATA ###############################
Folder containing data used in every project, specifically
# rat_hippocampus (Grosmark, A. D. & Buzsáki, 2016)
# https://crcns.org/data-sets/hc/hc-11/about-hc-11
both neural and behavioral data of 4 rats (achilles, buddy, cicero, gatsby) going back and forth a straight track 1.6m long. 
Data format and content:
- neural data are 2D matrices of spike counts per bin (25ms) x channels
- behavioral data are 2D matrices including position (continous) and direction (dummy 0/1) at every bin

# monkey_reaching_preload_smith_40 (macaque dataset, Chowdhury, Glaser, Miller, 2019)
# https://dandiarchive.org/dandiset/000127 
electrophysiological recordings for a monkey performing a 8 direction centre out reaching task with manipulandum. Both active and passive trials. Recordings are from -100ms to 500ms with 1ms time bins (smoothed with a gaussian kernel)
Data format and content
- neural data (spikes_active) are  EEG recordings (continuous) cast in a  2D matrix of size  Time*Channels (115800 bins x 65)...
    monkey_pos.neural.numpy()
    monkey_target.neural.numpy()
- Behavioural data - Active movement - (pos_active) is the arm position during the task; a 2D matrix of size Time*coordinates (115800 bins x 2)     
    monkey_pos.continuous_index.numpy()
- Active_target is the arm direction a Time*1 vector with labels 0-7 (8 directions)
    monkey_target.discrete_index.numpy()
- movement_dir=movement_dir_actpas count of trials 1-193 (0-192)
-vel_active:
-ephys:
-to_predict







