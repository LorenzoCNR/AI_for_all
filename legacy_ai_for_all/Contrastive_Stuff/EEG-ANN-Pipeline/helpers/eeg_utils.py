import numpy as np

def check_number_timepoints(trials):

    num_timepoints = None

    for trial in trials:
        if num_timepoints is None:
            num_timepoints = trial.num_timepoints
        else:
            if num_timepoints != trial.num_timepoints:
                num_timepoints = 'Trials of different length'
                break

    return num_timepoints


def get_label_format(label):

    if (type(label) is int) or (type(label) is np.int64) or (type(label) is np.int32) or (type(label) is np.int16)  or (type(label) is np.int8):
        return 'int'
    elif type(label) is str:
        return 'string'
    elif (type(label) is float) or (type(label) is np.float64) or (type(label) is np.float32):
        return 'float'
    elif isinstance(label, np.ndarray):
        return 'ndarray'
    else:
        raise ValueError(f'Formato label {type(label)} non riconosciuto')


#def check_labels_type(trials):

    # Capisco se è singola o multilabel
#    if type(trials[0].label) is dict:

#        labels_type = 'multi_label'
 #       labels_format = dict()

#        for key in trials[0].label:
#            labels_format[key] = get_label_format(trials[0].label[key])
#
 #   else:
#        labels_type = 'single_label'
  #      labels_format = get_label_format(trials[0].label)
#
 #   return labels_type, labels_format

def check_labels_type(trials):
    """
    Determina il tipo di etichette (single o multi-label) e il formato delle etichette.

    Args:
        trials (list): Lista di oggetti TrialEEG.

    Returns:
        tuple: (labels_type, labels_format)
    """
    # Ottieni la label del primo trial
    first_label = trials[0].labels

    if isinstance(first_label, dict):  # Controlla se è un dizionario
            if len(first_label) == 1:  # Una sola chiave, considerata single_label
                labels_type = 'single_label'
                labels_format = {list(first_label.keys())[0]: get_label_format(list(first_label.values())[0])}
            else:  # Più chiavi, considerata multi_label
                labels_type = 'multi_label'
                labels_format = {key: get_label_format(value) for key, value in first_label.items()}
    else:  # Non è un dizionario, considerata single_label
            labels_type = 'single_label'
            labels_format = get_label_format(first_label)

    return labels_type, labels_format


def convert_labels_string_to_int(labels):

    # Data una lista di labels sotto in formato stringa le converte in intero
    # e fornisce un dizionario per tornare indietro
    labels_int_to_str = dict()
    labels_str_to_int = dict()
    labels_converted = []

    for i, label in enumerate(labels):
        
        # Controllo se ho già trovato questa label
        if label not in labels_str_to_int:

            # Aggiungo un elemento a entrambi i dizionari di conversione
            new_label_int = len(labels_int_to_str)
            labels_str_to_int[label] = new_label_int
            labels_int_to_str[new_label_int] = label

        # La converto
        label = labels_str_to_int[label]
        labels_converted.append(label)

    return labels_converted, labels_int_to_str
