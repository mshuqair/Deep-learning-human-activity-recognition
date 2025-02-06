from keras.layers import Dense, Conv1D, Input, Dropout, MaxPooling1D, LSTM
from keras.initializers.initializers_v2 import GlorotUniform
from keras_tuner import BayesianOptimization, GridSearch
from keras.optimizers import Adam
from keras.regularizers import L2
from keras.models import Model
import numpy as np
import time
from data_processing import segment
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Optimizes model on data

def build_model(hp):
    pool_size = hp.Choice(name='pooling_size', values=[16])
    pool_strides = hp.Choice(name='pooling_strides', values=[4])

    padding_method = hp.Choice(name='padding_method', values=['same'])

    conv1_filters = hp.Choice(name='conv1_filters', values=[64])
    conv2_filters = hp.Choice(name='conv2_filters', values=[128])

    conv1_kernel = hp.Choice(name='conv1_kernel', values=[32])
    conv2_kernel = hp.Choice(name='conv2_kernel', values=[8])

    dense_units = hp.Choice(name='dense_units', values=[128])
    lstm_units = hp.Choice(name='lstm_units', values=[128])

    dropout_prob_1 = hp.Choice(name='dropout_prob_1', values=[0.1, 0.2, 0.3])
    dropout_prob_2 = hp.Choice(name='dropout_prob_2', values=[0.2, 0.3, 0.4])
    dropout_prob_3 = hp.Choice(name='dropout_prob_3', values=[0.3, 0.4, 0.5])
    dropout_prob_lstm = hp.Choice(name='dropout_prob_lstm', values=[0.1, 0.2, 0.3])
    kernel_reg = L2(name='kernel_reg', values=[0.0001])

    learning_rate = hp.Choice(name='learning_rate', values=[0.0001])
    optimizer = Adam(learning_rate=learning_rate)


    # input layer
    layer_input = Input(shape=data_shape, name='input_layer')

    # convolutional block layers
    layer_cnn_1_a = Conv1D(filters=conv1_filters, kernel_size=conv1_kernel, activation='relu', padding=padding_method,
                           kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(layer_input)
    layer_cnn_1_b = Conv1D(filters=conv1_filters, kernel_size=conv1_kernel, activation='relu', padding=padding_method,
                           kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(layer_cnn_1_a)
    layer_dropout_1 = Dropout(dropout_prob_1)(layer_cnn_1_b)
    layer_pooling_1 = MaxPooling1D(pool_size=pool_size, strides=pool_strides)(layer_dropout_1)

    layer_cnn_2_a = Conv1D(filters=conv2_filters, kernel_size=conv2_kernel, activation='relu', padding=padding_method,
                           kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(layer_pooling_1)
    layer_cnn_2_b = Conv1D(filters=conv2_filters, kernel_size=conv2_kernel, activation='relu', padding=padding_method,
                           kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(layer_cnn_2_a)
    layer_dropout_2 = Dropout(dropout_prob_2)(layer_cnn_2_b)

    layer_lstm = LSTM(units=lstm_units, kernel_initializer=kernel_init, dropout=dropout_prob_lstm)(layer_dropout_2)
    layer_final = layer_lstm

    # classification layers
    layer_dense_1 = Dense(units=dense_units, activation='relu', kernel_initializer=kernel_init,
                          kernel_regularizer=kernel_reg)(layer_final)
    layer_dropout_3 = Dropout(dropout_prob_3)(layer_dense_1)
    layer_output = Dense(units=num_classes, activation=None, name='output_layer')(layer_dropout_3)

    # construct the model
    model_cnn = Model(inputs=layer_input, outputs=layer_output)

    # compile and model summary
    model_cnn.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model_cnn


# Main Code
start_time = time.time()

# Load data file
data = np.load(file='data/MHEALTH Original.npz')
x_raw = data['x']  # 3D wrist acc and 3D ankle acc data of all subjects concatenated
y_raw = data['y']  # labels
pNo = data['subject_index'] + 1  # participant number

# Signal parameters
fs = 50  # sampling frequency of the original signal
window_width = 5  # window size in seconds
window_length = int(fs * window_width)  # window size in samples
overlap = 0  # overlap rate between windows, => 4s ==> rate = 4/5 = 0.8
numChannels = x_raw.shape[1]  # 6 for 3D wrist and ankle sensors

# Tuning settings
epochs = 15
batch_size = 32

# Model parameters and setup
kernel_init = GlorotUniform(seed=1)
data_shape = (window_length, numChannels)
num_classes = len(np.unique(y_raw))

# Segment the signal
x_raw, y_raw = segment(features=x_raw, targets=y_raw,
                       window_size=window_length, num_cols=numChannels, overlap_rate=overlap)

# Split the data into training-validation
# Split percentage-wise
x_train, x_valid, y_train, y_valid = train_test_split(x_raw, y_raw, test_size=0.2, shuffle=True)

# Split subject-wise
# sub_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
# sub_valid = np.array([10])
# x_train = x_raw[np.isin(pNo, sub_train)]
# y_train = y_raw[np.isin(pNo, sub_train)]
# x_valid = x_raw[np.isin(pNo, sub_valid)]
# y_valid = y_raw[np.isin(pNo, sub_valid)]


# calculate training scaling info
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_valid = scaler.transform(x_valid)


# Optimize model
# Using Bayesian Optimization method
tuner = BayesianOptimization(hypermodel=build_model, objective='val_loss', max_trials=20,
                             directory='model optimization', project_name='cnn_lstm_har', overwrite=True)

# Using Grid Search method
# tuner = GridSearch(hypermodel=build_model, objective='val_loss',
#                    max_trials=None,
#                    directory='model optimization', project_name='cnn_lstm_har', overwrite=True)


print('Summary of the search space:')
tuner.search_space_summary()
tuner.search(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_valid, y_valid))

# summary of results
print('Summary of the search results:')
tuner.results_summary()

# best hyperparameters
print(print('Best hyperparameters:'))
best_hp = tuner.get_best_hyperparameters()[0]
print(best_hp.values)

# calculate and print elapsed time
elapsed_time = format((time.time() - start_time) / 60, '.2f')
print("Elapsed time: " + str(elapsed_time) + " minutes")
