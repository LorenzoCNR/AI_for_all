# Decoding Pipeline using KNN


### Main Module: `valid_funcs_KNN`
This is the core module that controls the decoding process (the flow between the submodules and the main functions)

### Submodules:
- **h5_management** (external module): A collection of functions for saving,loading and exploring data to/from `.h5` files.
- **split_data_fncts** (external module): Splits data based on trials into train, validation, and test sets, or further splits the train set into subtrain and validation subsets.
- **model_utils** (external module): Used to set up and run different dimensionality reduction models, including CEBRA, UMAP, t-SNE, and others.

### Internal Functions:

#### `load_params`:
Loads parameters from a YAML file, which are used to configure the models and control the training process.
The YAML file contains parameters for each model that need to be tuned. Some parameters are fixed, while others are part of a grid and should be rotated for optimization; if you're not grid searching, the first value of a grid will be the default one


#### `process_data`:
- Iterates over the parameters and trains the model.
- Generates embeddings for the provided sets (train, subtrain, validation).
- Uses subtrain and validation embeddings to find the best `k` for KNN (based on \(R^2\) fit score).
- Once the best `k` is chosen, the function defines a KNN regressor and classifier built on the subtrain embeddings and validates them on the validation set to check the predicted distances.
- The model with the minimum error is saved in `.pth` format (PyTorch), and for each subject (rat), a model and an HDF5 file with optimal parameter values is generated.

## MAIN Module, `valid_funcs_KNN`  iterates over a list of subjects (rats) and applies the decoding pipeline to each one. It requires the following inputs (call from command...order is not taken in account, just the flag):

python valid_funcs.py --input_dir <path_input> --output_dir <path_output> --param <file_param.yaml> --model_type <model type> --use_grid --replace --subtrain --rat_list achilles cicero


- **Input and output directories**: Paths where the data and results are stored, as strings.
- **`rat_list`**: (Optional) A list of subjects to process. If you have a dictionary with many subjects, you can subset them. This is list...
- **`param.yaml` file**: The YAML file containing the model parameters, provided as a string.
- **Model choice**: The name of the model (e.g., CEBRA, UMAP, t-SNE), provided as a string.
- **Grid search option**: Whether or not to use grid search for hyperparameters, provided as a boolean (True/False).
- **Replace existing models**: Whether or not to replace existing saved models in case they already exist, provided as a boolean (True/False).
-** subtrain: add a further split to the train set, boolean (True/False)

### Flow of the function:
1. **Data Loading**: The function first loads the data for the specified rats from the input directory.
    Just notice that the function loads jl file and the name is specified in the main function
2. **Data Splitting**: It then splits the data into train, validation, and test sets, or further splits the train set into subtrain and validation sets.(choice between regular, and subtrain)
Just notice that if subtrain is False 
3. **Model Training**: Using the specified model, the function trains and evaluates the model on the subtrain and validation sets to find the best `k` for KNN. Given the best k we set a KNN regressors and a KNN classifier in order compute the prediction error given the subtrain and validation sets
4. **Model Saving**: The model with the best performance is saved as a `.pth` file, and the optimal parameters are stored in an HDF5 file for each subject.


### Dependencies (req.txt): to be done
### Contributions (external): to be done
### Improvements: add to the parser also the data (call from command)