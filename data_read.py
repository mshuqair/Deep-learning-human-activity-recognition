import numpy as np

# Load data file
data = np.load(file='data/mhealth.npz')

x = data['x']  # 3D wrist acc and 3D ankle acc data of all subjects concatenated
y = data['y']  # labels
subject_index = data['subject_index'] + 1

print(x.shape)
print(y.shape)
print(subject_index.shape)

print(np.unique(y))
print(np.unique(subject_index))
