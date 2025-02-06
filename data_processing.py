from scipy.interpolate import CubicSpline
from transforms3d.axangles import axangle2mat
import numpy as np
import math


# Several data processing functions

def mean_std(data, axis=0):
    mean = np.mean(data, axis=axis)
    std = np.std(data, axis=axis)
    return mean, std


def next_pow_2(n):
    # Calculate log2 of N
    a = int(math.log2(n))
    if 2 ** a == n:  # If 2^a is equal to N, return N
        return n
    return 2 ** (a + 1)  # Return 2^(a + 1)


def windowing(data, size, overlap_rate):
    start = 0  # starting index
    while start < (len(data) - size):  # iterate through all timestamps
        yield start, start + size  # returns the start and stop indices for a given window
        start += int(size * (1 - overlap_rate))  # uses a step size


# This the main function for data segmentation
def segment(features, targets, window_size, num_cols, overlap_rate):
    step_size = int(window_size * (1 - overlap_rate))

    # Number of segmented windows follows this formula: int(number of total data timestamps / step size)
    segments = np.zeros((int(len(features) / step_size), window_size, num_cols))
    labels = np.zeros((int(len(targets) / step_size)))  # int(number of timestamps / step size)

    i_segment = 0
    i_label = 0

    for (start, end) in windowing(features, window_size, overlap_rate):
        if (len(features[start:end]) == window_size) and (i_segment != len(segments)) and (i_label != len(labels)):
            if len(np.unique(targets[start:end])) == 1:
                segments[i_segment] = features[start:end]
                labels[i_label] = np.unique(targets[start:end])[0]
                i_label += 1
                i_segment += 1

    # Remove empty windows resulted from segmentation
    if i_segment < segments.shape[0]:
        segments = segments[:i_segment]
        labels = labels[:i_label]

    return segments, labels


# Data augmentation functions
# This function returns the original signal
def da_original(x):
    return x


# Adding noise
def da_jitter(x, sigma=0.05):
    my_noise = np.random.normal(loc=0, scale=sigma, size=x.shape)
    return x + my_noise


# Apply scaling
def da_scaling(x, sigma=0.1):
    scaling_factor = np.random.normal(loc=1.0, scale=sigma, size=(1, x.shape[1]))
    my_noise = np.matmul(np.ones((x.shape[0], 1)), scaling_factor)
    return x * my_noise


# This example using cubic splice is not the best approach to generate random curves.
# You can use other approaches, e.g., Gaussian process regression, Bézier curve, etc.
def generate_random_curves(x, sigma, knot):
    # Random curves around 1.0
    xx = (np.ones((x.shape[1], 1)) * (np.arange(0, x.shape[0], (x.shape[0] - 1) / (knot + 1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, x.shape[1]))
    x_range = np.arange(x.shape[0])
    cs_x = CubicSpline(xx[:, 0], yy[:, 0])
    cs_y = CubicSpline(xx[:, 1], yy[:, 1])
    cs_z = CubicSpline(xx[:, 2], yy[:, 2])
    return np.array([cs_x(x_range), cs_y(x_range), cs_z(x_range)]).transpose()


# Apply Magnitude Warping
def da_mag_warp(x, sigma=0.2, knot=4):
    return x * generate_random_curves(x, sigma, knot)


def distort_time_steps(x, sigma, knot):
    tt = generate_random_curves(x, sigma, knot)  # Regard these samples around 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)  # Add intervals to make a cumulative graph
    # Make the last value to have x.shape[0]
    t_scale = [(x.shape[0] - 1) / tt_cum[-1, 0], (x.shape[0] - 1) / tt_cum[-1, 1], (x.shape[0] - 1) / tt_cum[-1, 2]]
    tt_cum[:, 0] = tt_cum[:, 0] * t_scale[0]
    tt_cum[:, 1] = tt_cum[:, 1] * t_scale[1]
    tt_cum[:, 2] = tt_cum[:, 2] * t_scale[2]
    return tt_cum


# Apply time warping
def da_time_warp(x, sigma=0.2, knot=4):
    tt_new = distort_time_steps(x, sigma, knot)
    x_new = np.zeros(x.shape)
    x_range = np.arange(x.shape[0])
    x_new[:, 0] = np.interp(x_range, tt_new[:, 0], x[:, 0])
    x_new[:, 1] = np.interp(x_range, tt_new[:, 1], x[:, 1])
    x_new[:, 2] = np.interp(x_range, tt_new[:, 2], x[:, 2])
    return x_new


# Apply rotation
def da_rotation(x):
    axis = np.random.uniform(low=-1, high=1, size=x.shape[1])
    angle = np.random.uniform(low=-np.pi, high=np.pi)
    return np.matmul(x, axangle2mat(axis, angle))


# Apply permutation
def da_permutation(x, n_perm=4, min_seg_length=100):
    x_new = np.zeros(x.shape)
    idx = np.random.permutation(n_perm)
    b_while = True
    segments = None
    while b_while:
        segments = np.zeros(n_perm + 1, dtype=int)
        segments[1:-1] = np.sort(np.random.randint(min_seg_length, x.shape[0] - min_seg_length, n_perm - 1))
        segments[-1] = x.shape[0]
        if np.min(segments[1:] - segments[0:-1]) > min_seg_length:
            b_while = False
    pp = 0
    for j in range(n_perm):
        x_temp = x[segments[idx[j]]:segments[idx[j] + 1], :]
        x_new[pp:pp + len(x_temp), :] = x_temp
        pp += len(x_temp)
    return x_new
