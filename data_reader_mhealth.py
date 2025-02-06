import numpy as np


info = {
    'subject': [],
    'data_len': []
}

dataframes = []

# Iterate through files
for i in range(1, 11):

    subject = 'subject' + str((i))

    print(f'\nReading file {i} of 10')
    print(f'filename: mhealth_subject{i}.log')

    data = np.loadtxt(f'data/MHEALTHDATASET/mhealth_subject{i}.log')

    # Remove unnecessary columns
    data = np.delete(data, [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 17, 18, 19, 20, 21, 22], axis=1)

    # Remove null class data
    data = np.delete(data, np.where(data[:, 6] == 0), axis=0)

    # Remove rows with NaN, if present
    data = data[~np.isnan(data).any(axis=1)]

    # PREPARE DATA TO SIMULATE NON-CONTINUOUS RECORDING
    # Separate inputs and labels into separate arrays
    inputs_og = np.delete(data, 6, 1)
    labels_og = data[:, 6].reshape((data[:, 6].shape[0], 1))

    # Remove 5s (250 timestamps) of activity at beginning and end of each labeled activity to simulate non-continuous
    # recording
    # https://stackoverflow.com/questions/28242032/track-value-changes-in-a-repetitive-list-in-python
    # Detect at what indices labels change
    change_indices = [i for i in range(1, len(labels_og)) if labels_og[i] != labels_og[i - 1]]
    # Create array of all indices to remove
    remove_indices = []
    for i in change_indices:
        for j in reversed(range(1, 251)):
            if i - j > len(inputs_og):  # Ensures index does not go out of range
                remove_indices.append(i - j)
        for k in range(250):
            if i + k < len(inputs_og):  # Ensures index does not go out of range
                remove_indices.append(i + k)
    # Remove those indices from array row-wise
    inputs_new = np.delete(inputs_og, remove_indices, axis=0)
    labels_new = np.delete(labels_og, remove_indices, axis=0)

    info['data_len'].append(inputs_new.shape[0])
    info['subject'].append(subject)

    df = np.hstack((inputs_new, labels_new))

    dataframes.append(df)

print('\n*** DONE ***')

#  EXTRACT NECESSARY INFORMATION ON WHERE EACH SUBJECT'S DATA IS IN THE COMBINED DATA FOR LOSO KFOLD CV
df_full = np.concatenate(dataframes)

indices = {
    'subject': info['subject'],  # keep same reference to other list so subject info stays consistent
    'start': [],
    'stop': []
}

cumsum = np.cumsum(info['data_len'])

for i in range(len(info['data_len'])):
    if i == 0:
        indices['start'].append(0)
    else:
        indices['start'].append(cumsum[i - 1])
    indices['stop'].append(indices['start'][i] + info['data_len'][i] - 1)

folds = np.transpose(np.array([indices['subject'], indices['start'], indices['stop']]))

# Separate inputs and labels into separate arrays
inputs_final = np.delete(df_full, 6, axis=1)
labels_final = df_full[:, 6].reshape((df_full[:, 6].shape[0], 1))

#  DATA PREP FOR MODEL TRAINING AND K-FOLD CV

# Create new subject_index array to store updated locations of start:stop indices for subjects after
# deleting/modifying data
subject_index = np.zeros(labels_final.shape)
for i in range(len(folds)):
    start = int(folds[i][1])
    stop = int(folds[i][2])
    subject_index[start:stop] = i

# Remove labels not common with PAMAP2
del_indices = np.where(
    (labels_final == 6) | (labels_final == 7) | (labels_final == 8) | (labels_final == 10) | (labels_final == 12))
y = np.delete(labels_final, del_indices)
x = np.delete(inputs_final, del_indices, axis=0)
subject_index = np.delete(subject_index, del_indices)

# One-hot encoding
labels = np.unique(y)
for j in range(7):
    np.place(y, y == labels[j], j)

np.savez(file='data/mhealth.npz', x=x, y=y, folds=folds, subject_index=subject_index)
