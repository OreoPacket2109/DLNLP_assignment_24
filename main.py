#====================|Importing Dependencies|====================
import numpy as np
import pandas as pd
from keras.layers import Embedding, MultiHeadAttention, Concatenate, Reshape
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

#User-defined classes
from corpus import corpus
from lstm import lstm
from cnn import cnn
from cnn_with_lstm import cnn_with_lstm
from lstm_with_attention import lstm_with_attention

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Attention
from tensorflow.keras.models import Model
from keras import Sequential, Input
import seaborn as sns

#====================|1. Importing Datasets|====================

#Importing first dataset
dir_a = "Datasets/labeled_data.csv"
df_a = pd.read_csv(dir_a)
dataset_a = df_a['tweet']
labels_a = df_a['class']

# #Importing second dataset
# dir_b = "Datasets/additional_data.csv"
# df_b = pd.read_csv(dir_b)
# dataset_b = df_b[df_b['label'] == 'hate']
# dataset_b = dataset_b['text']
# labels_b = pd.Series(np.zeros(len(dataset_b)))
# dataset_b.reset_index(drop = True, inplace = True)

#Combining the two datasets
# combined_dataset = pd.concat([dataset_a, dataset_b], ignore_index=True)
# labels_b.reset_index(drop = True, inplace = True)
# combined_labels = pd.concat([labels_a, labels_b], ignore_index=True)

print(labels_a.shape)
# print(labels_b.shape)
# print(combined_labels.shape)

#====================|2. Dataset Analysis|====================
my_dataset = corpus(dataset_a, labels_a)
my_dataset.show_class_distribution()
my_dataset.show_tweet_length_distribution()

#====================|3. Comparing Model Performances|====================

#====================|3.1. LSTM|====================
# my_lstm = lstm(my_dataset.X_train, my_dataset.y_train, my_dataset.X_val, my_dataset.y_val, my_dataset.X_test, my_dataset.y_test, 3, my_dataset.max_length)
# my_lstm.train(epoch = 1, batch_size = 32)
# my_lstm.test(my_dataset.tokenizer)

#====================|3.2. CNN|====================
# my_cnn = cnn(my_dataset.X_train, my_dataset.y_train, my_dataset.X_val, my_dataset.y_val, my_dataset.X_test, my_dataset.y_test, 3, my_dataset.max_length)
# my_cnn.train(epoch = 10, batch_size = 32)
# my_cnn.test()

#====================|3.3. Hybrid Model|====================
my_hybrid = cnn_with_lstm(my_dataset.X_train, my_dataset.y_train, my_dataset.X_val, my_dataset.y_val, my_dataset.X_test, my_dataset.y_test, my_dataset.max_length, embedding_dim = 128, vocab_size = my_dataset.vocab_size, num_classes = 3, n_filters = 64, kernel_size = 7, pool_size = 2, n_lstm = 128, dropout = 0.2, recurrent_dropout= 0.2)
my_hybrid.train(epoch = 3, batch_size = 64)
my_hybrid.test()

my_lstm_with_attention = lstm_with_attention(my_dataset.X_train, my_dataset.y_train, my_dataset.X_val, my_dataset.y_val, my_dataset.X_test, my_dataset.y_test, my_dataset.max_length, my_dataset.vocab_size, embedding_dim = 128, lstm_units = 64, dense_units = 32, num_classes = 3, dropout = 0.2)
my_lstm_with_attention.train(epoch = 3, batch_size = 64)
my_lstm_with_attention.test(tokenizer = my_dataset.tokenizer)

