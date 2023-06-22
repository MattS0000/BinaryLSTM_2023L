import tensorflow as tf
from tensorflow.keras.models import  Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GlobalMaxPooling1D, Bidirectional
import BinaryLSTM

def get_small(optimizer='adam',lr = 0.001, binary=False, n_units=128, max_len = 80, vocabulary_size = 10000 ,embed_size = 32):
    model = Sequential()
    model.add(Embedding(vocabulary_size,embed_size,input_length = max_len))
    if binary:
        model.add(BinaryLSTM.BinaryLSTM(n_units, return_sequences=True))
    else:
        model.add(LSTM(n_units, return_sequences=True))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.25))
    model.add(Dense(4, activation='softmax'))
    if optimizer=="adam":
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        opt = tf.keras.optimizers.RMSprop(learning_rate=lr)
    model.compile(loss = 'sparse_categorical_crossentropy',
                optimizer = opt,
                metrics = ['accuracy']             
                )
    return model

def get_medium(optimizer='adam',lr = 0.001, binary=False, n_units=128, max_len = 80, vocabulary_size = 10000 ,embed_size = 32):
    model = Sequential()
    model.add(Embedding(vocabulary_size,embed_size,input_length = max_len))
    if binary:
        model.add(BinaryLSTM.BinaryLSTM(n_units, return_sequences=True))
    else:
        model.add(LSTM(n_units, return_sequences=True))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu')) 
    model.add(Dropout(0.25))
    model.add(Dense(4, activation='softmax')) 
    if optimizer=="adam":
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        opt = tf.keras.optimizers.RMSprop(learning_rate=lr)
    model.compile(loss = 'sparse_categorical_crossentropy',
                optimizer = opt,
                metrics = ['accuracy']             
                )
    return model

def get_large(optimizer='adam',lr = 0.001, binary=False, n_units=128, max_len = 80, vocabulary_size = 10000 ,embed_size = 32):
    model = Sequential()
    model.add(Embedding(vocabulary_size,embed_size,input_length = max_len))
    if binary:
        model.add(Bidirectional(BinaryLSTM.BinaryLSTM(n_units, return_sequences=True)))
        model.add(Bidirectional(BinaryLSTM.BinaryLSTM(int(n_units/2), return_sequences=True)))
    else:
        model.add(Bidirectional(LSTM(n_units, return_sequences=True)))
        model.add(Bidirectional(LSTM(int(n_units/2), return_sequences=True)))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(4, activation='softmax'))
    if optimizer=="adam":
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        opt = tf.keras.optimizers.RMSprop(learning_rate=lr)
    model.compile(loss = 'sparse_categorical_crossentropy',
                optimizer = optimizer,
                metrics = ['accuracy']             
                )
    return model