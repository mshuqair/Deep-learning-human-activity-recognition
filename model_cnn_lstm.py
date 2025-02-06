from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.layers import Input, Dense, Dropout, Conv1D, MaxPooling1D, LSTM
from keras.optimizers.optimizer_v2.adam import Adam
from keras.models import Model
from keras.initializers.initializers_v2 import GlorotUniform
import numpy as np
import time
import os
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from data_processing import segment
import pickle


def func_model():
    # model parameters
    optimizer = Adam(learning_rate=learning_rate)
    dropout_prob_1 = 0.1
    dropout_prob_2 = 0.2
    dropout_prob_3 = 0.3
    dropout_prob_lstm = 0.2

    # input / output setup
    input_shape = (window_length, channels_num)
    output_shape = classes_num

    # classification layers
    dense_units = 128

    # LSTM units
    lstm_units = 128

    # conv. blocks hyperparameters
    padding_method = 'same'
    conv1_filters = 64
    conv2_filters = 128
    conv1_kernel = 32  # 32
    conv2_kernel = 8  # 8
    pool_size = 16  # 16
    pool_strides = 4  # 4

    # input layer
    layer_input = Input(shape=input_shape)

    # convolutional block layers
    layer_cnn_1_a = Conv1D(filters=conv1_filters, kernel_size=conv1_kernel, activation='relu', padding=padding_method,
                           kernel_initializer=kernel_initializer)(layer_input)
    layer_cnn_1_b = Conv1D(filters=conv1_filters, kernel_size=conv1_kernel, activation='relu', padding=padding_method,
                           kernel_initializer=kernel_initializer)(layer_cnn_1_a)
    layer_dropout_1 = Dropout(dropout_prob_1)(layer_cnn_1_b)
    layer_pooling_1 = MaxPooling1D(pool_size=pool_size, strides=pool_strides)(layer_dropout_1)

    layer_cnn_2_a = Conv1D(filters=conv2_filters, kernel_size=conv2_kernel, activation='relu', padding=padding_method,
                           kernel_initializer=kernel_initializer)(layer_pooling_1)
    layer_cnn_2_b = Conv1D(filters=conv2_filters, kernel_size=conv2_kernel, activation='relu', padding=padding_method,
                           kernel_initializer=kernel_initializer)(layer_cnn_2_a)
    layer_dropout_2 = Dropout(dropout_prob_2)(layer_cnn_2_b)

    layer_lstm = (LSTM(units=lstm_units, kernel_initializer=kernel_initializer, dropout=dropout_prob_lstm)
                  (layer_dropout_2))
    layer_final = layer_lstm

    # classification layers
    layer_dense_1 = Dense(units=dense_units, activation='relu', kernel_initializer=kernel_initializer)(layer_final)
    layer_dropout_3 = Dropout(dropout_prob_3)(layer_dense_1)
    layer_output = Dense(units=output_shape, activation='softmax', kernel_initializer=kernel_initializer)(layer_dropout_3)

    # construct the model
    model = Model(inputs=layer_input, outputs=layer_output)

    # compile and model summary
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model


def scheduler(epoch, lr):
    if epoch <= (0.1 * epochs):
        lr_new = lr
    else:
        lr_new = lr - lr_decay
    return lr_new


# Main code
start_time = time.time()

# results parameters
save_history = True  # whether to save the training history in pickle file or not
save_predictions = True  # whether to save the model predictions
save_dir = './output/' + os.path.basename(__file__)[0:-3] + '/'
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

# training settings
epochs = 35
batch_size = 32
learning_rate = 0.0001
lr_decay = learning_rate / epochs
kernel_initializer = GlorotUniform(seed=1)

# Import the raw data
data = np.load(file='data/mhealth.npz')
x_raw = data['x']  # 3D wrist acc and 3D ankle acc data of all subjects concatenated
y_raw = data['y']  # labels
p_num = data['subject_index'] + 1  # participant number)

# Signal parameters
fs = 50  # sampling frequency of the original signal
window_width = 5  # window size in seconds
window_length = int(fs * window_width)  # window size in samples
overlap = 0.5  # overlap rate between windows, => 4s ==> rate = 4/5 = 0.8
channels_num = x_raw.shape[1]  # 6 for 3D wrist and ankle sensors
classes_num = len(np.unique(y_raw))

# Split data into training test
p_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
p_test = np.array([10])

x_train = x_raw[np.isin(p_num, p_train)]
y_train = y_raw[np.isin(p_num, p_train)]

x_test = x_raw[np.isin(p_num, p_test)]
y_test = y_raw[np.isin(p_num, p_test)]

# Segmenting
x_train, y_train = segment(features=x_train, targets=y_train,
                           window_size=window_length, num_cols=channels_num, overlap_rate=overlap)
x_test, y_test = segment(features=x_test, targets=y_test,
                         window_size=window_length, num_cols=channels_num, overlap_rate=overlap)

# Prepare for training
# Split data
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Scaling
mean = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0)
x_train = (x_train - mean) / std
x_valid = (x_valid - mean) / std
x_test = (x_test - mean) / std

y_train = to_categorical(y_train, num_classes=classes_num)
y_valid = to_categorical(y_valid, num_classes=classes_num)

# Creating required model callbacks
callback_scheduler = LearningRateScheduler(scheduler)
callback_checkpoint = ModelCheckpoint(filepath=save_dir + 'model_cnn_best.h5', monitor='val_loss',
                                      save_best_only=True, save_freq='epoch')

model_har = func_model()
history = model_har.fit(x=x_train, y=y_train,
                        validation_data=(x_valid, y_valid),
                        batch_size=batch_size, epochs=epochs,
                        callbacks=[callback_scheduler, callback_checkpoint],
                        verbose=1)

# Save the training history
if save_history:
    model_history = history.history
    with open(save_dir + 'model_train_history.pkl', 'wb') as file:
        pickle.dump(model_history, file)


# Calculate model predictions
print('Testing metrics...')
y_predicted = model_har.predict(x_test)
y_predicted = np.argmax(y_predicted, axis=1)

if save_predictions:
    model_predictions = [y_test, y_predicted, p_test]
    with open(save_dir+'model_predictions.pkl', 'wb') as file:
        pickle.dump(model_predictions, file)


# calculate and print elapsed time
elapsed_time = format((time.time() - start_time) / 60, '.2f')
print("Elapsed time: " + str(elapsed_time) + " minutes")
