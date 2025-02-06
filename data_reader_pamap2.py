import numpy as np

info = {
    'subject': [],
    'data_len': []
}

dataframes = []

# Iterate through files
for i in range(1, 9):
    subject = 'subject' + str((i))

    print(f'\nReading file {i} of 8')
    print(f'filename: subject10{i}.dat')

    data = np.loadtxt(f'data/PAMAP2 Physical Activity Monitoring Dataset/Protocol/subject10{i}.dat')

    # Remove unnecessary columns
    data = np.delete(data,
                     [0, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                      30, 31, 32, 33, 34, 35, 36, 37, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53], axis=1)

    # Remove null class data
    data = np.delete(data, np.where(data[:, 0] == 0), axis=0)

    # Remove rows with NaN, if present
    data = data[~np.isnan(data).any(axis=1)]

    # Downsample from 100Hz -> 50Hz
    data = np.delete(data, list(range(0, data.shape[0], 2)), axis=0)

    # For PAMAP2, rearrange the columns so the order matches that of MHEALTH
    new_col_order = [4, 5, 6, 1, 2, 3, 0]
    data = data[:, new_col_order]

    ##### PREPARE DATA TO SIMULATE NON-CONTINUOUS RECORDING #####
    # Separate inputs and labels into seperate arrays
    inputs_og = np.delete(data, 6, 1)
    labels_og = data[:, 6].reshape((data[:, 6].shape[0], 1))

    # Remove 5s (250 timestamps) of activity at beginning and end of each labeled activity to simulate non-continuous recording
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

##### EXTRACT NECESSARY INFORMATION ON WHERE EACH SUBJECT'S DATA IS IN THE COMBINED DATA FOR LOSO KFOLD CV#####
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

# Separate inputs and labels into seperate arrays
inputs_final = np.delete(df_full, 6, axis=1)
labels_final = df_full[:, 6].reshape((df_full[:, 6].shape[0], 1))

##### DATA PREP FOR MODEL TRAINING AND KFOLD CV #####

# Create new subject_index array to store updated locations of start:stop indices for subjects after deleting/modifying data
subject_index = np.zeros(labels_final.shape)
for i in range(len(folds)):
    start = int(folds[i][1])
    stop = int(folds[i][2])
    subject_index[start:stop] = i

# Remove labels not common with MHEALTH
del_indices = np.where(
    (labels_final == 7) | (labels_final == 13) | (labels_final == 16) | (labels_final == 17) | (labels_final == 24))
y = np.delete(labels_final, del_indices)
x = np.delete(inputs_final, del_indices, axis=0)
subject_index = np.delete(subject_index, del_indices)

# Rename labels of common activities to correspond to mhealth
old_labels = [1, 2, 3, 4, 5, 6, 12]  # original pamap2 labels
new_labels = [3, 2, 1, 4, 11, 9, 5]  # corresponding mhealth labels

y_new = np.copy(y)
for i in range(len(old_labels)):
    y_new[y == old_labels[i]] = new_labels[i]

# Rename labels from 0, num_labels-1 to prepare data for one-hot encoding
labels = np.array([1, 2, 3, 4, 5, 9, 11])
for j in range(7):
    np.place(y_new, y_new == labels[j], j)

# np.savez('../../har models/data/PAMAP2 Original.npz', x=x, y=y_new, folds=folds, subject_index=subject_index)
np.savez('data/pamap.npz', x=x, y=y_new, folds=folds, subject_index=subject_index)
