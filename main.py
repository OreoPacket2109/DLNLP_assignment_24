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
header = df_a.columns.tolist()
print(header)
dataset_a = df_a['tweet']
labels_a = df_a['class']

# print(type(dataset_a))

# #Importing second dataset
# dir_b = "Datasets/additional_data.csv"
# df_b = pd.read_csv(dir_b)
# dataset_b = df_b[df_b['label'] == 'hate']
# dataset_b = dataset_b['text']
# labels_b = pd.Series(np.zeros(len(dataset_b)))
# dataset_b.reset_index(drop = True, inplace = True)
#
# #Combining the two datasets
# combined_dataset = pd.concat([dataset_a, dataset_b], ignore_index=True)
# labels_b.reset_index(drop = True, inplace = True)
# combined_labels = pd.concat([labels_a, labels_b], ignore_index=True)
#
# print(labels_a.shape)
# print(labels_b.shape)
# print(combined_labels.shape)

#====================|2. Dataset Analysis|====================
my_dataset = corpus(dataset_a, labels_a)
my_dataset.show_old_distribution()
my_dataset.show_class_distribution()
my_dataset.show_tweet_length_distribution()
#====================|3. Comparing Model Performances|====================

#====================|3.1. LSTM|====================
my_lstm = lstm(my_dataset.X_train, my_dataset.y_train, my_dataset.X_val, my_dataset.y_val, my_dataset.X_test, my_dataset.y_test, 3, my_dataset.max_length, n_lstm = 64, dropout = 0.2, recurrent_dropout = 0.2)
my_lstm.train(epoch = 1, batch_size = 32)
my_lstm.test(my_dataset.tokenizer)
print("====================")
print("LSTM F1:")
print(my_lstm.f1)
print("LSTM Accuracy:")
print(my_lstm.accuracy)

#====================|3.2. CNN|====================
my_cnn = cnn(my_dataset.X_train, my_dataset.y_train, my_dataset.X_val, my_dataset.y_val, my_dataset.X_test, my_dataset.y_test, 3, my_dataset.max_length, n_filter = 128, kernel_size= 4, pool_size= 8)
my_cnn.train(epoch = 5, batch_size = 32)
my_cnn.test()
print("====================")
print('CNN F1:')
print(my_cnn.f1)
print("CNN: Accuracy:")
print(my_cnn.accuracy)
#====================|3.3. Hybrid Model|====================

#Testing hyperparameters

#Kernel size
# kernel_sizes = [2, 4, 8, 16]
# losses = np.zeros(len(kernel_sizes))
#
# for i in range(len(kernel_sizes)):
#     my_hybrid = cnn_with_lstm(my_dataset.X_train, my_dataset.y_train, my_dataset.X_val, my_dataset.y_val,
#                               my_dataset.X_test, my_dataset.y_test, my_dataset.max_length, embedding_dim=128,
#                               vocab_size=my_dataset.vocab_size, num_classes=3, n_filters=32, kernel_size=kernel_sizes[i],
#                               pool_size=4, n_lstm=32, dropout=0.2, recurrent_dropout=0.2)
#     my_hybrid.model.summary()
#
#     my_hybrid.train(epoch=2, batch_size=32)
#     val_loss = my_hybrid.get_val_loss()
#     losses[i] = val_loss
#
# print(losses)
# plt.plot(kernel_sizes, losses)
# plt.grid(True)
# plt.xlabel("Kernel Size")
# plt.ylabel("Validation Loss")
# plt.title("Validation Loss vs Kernel Size")
# plt.show()

# #Number of convoluton filters
# n_filters = [2, 4, 8, 16, 32, 64, 128, 256]
# losses = np.zeros(len(n_filters))
#
# for i in range(len(n_filters)):
#     my_hybrid = cnn_with_lstm(my_dataset.X_train, my_dataset.y_train, my_dataset.X_val, my_dataset.y_val,
#                               my_dataset.X_test, my_dataset.y_test, my_dataset.max_length, embedding_dim=128,
#                               vocab_size=my_dataset.vocab_size, num_classes=3, n_filters=n_filters[i], kernel_size=4,
#                               pool_size=4, n_lstm=32, dropout=0.2, recurrent_dropout=0.2)
#     my_hybrid.model.summary()
#
#     my_hybrid.train(epoch=2, batch_size=32)
#     val_loss = my_hybrid.get_val_loss()
#     losses[i] = val_loss
#
# print(losses)
# plt.plot(n_filters, losses)
# plt.grid(True)
# plt.xlabel("Number of Filters")
# plt.ylabel("Validation Loss")
# plt.title("Validation Loss vs Number of Filters")
# plt.show()

#Pooling layer size
# pool_size = [2, 3, 4, 5, 7, 8, 10]
# losses = np.zeros(len(pool_size))
#
# for i in range(len(pool_size)):
#     my_hybrid = cnn_with_lstm(my_dataset.X_train, my_dataset.y_train, my_dataset.X_val, my_dataset.y_val,
#                               my_dataset.X_test, my_dataset.y_test, my_dataset.max_length, embedding_dim=128,
#                               vocab_size=my_dataset.vocab_size, num_classes=3, n_filters=128, kernel_size=4,
#                               pool_size=pool_size[i], n_lstm=64, dropout=0.2, recurrent_dropout=0.2)
#     my_hybrid.model.summary()
#
#     my_hybrid.train(epoch=2, batch_size=32)
#     val_loss = my_hybrid.get_val_loss()
#     losses[i] = val_loss
#
# print(losses)
# plt.plot(pool_size, losses)
# plt.grid(True)
# plt.xlabel("Pool Size")
# plt.ylabel("Validation Loss")
# plt.title("Validation Loss vs. Pool Size")
# plt.show()


#Number of LSTM units
# n_lstm = [2, 4, 8, 16, 32, 64, 128, 256]
# losses = np.zeros(len(n_lstm))
#
# for i in range(len(n_lstm)):
#     my_hybrid = cnn_with_lstm(my_dataset.X_train, my_dataset.y_train, my_dataset.X_val, my_dataset.y_val,
#                               my_dataset.X_test, my_dataset.y_test, my_dataset.max_length, embedding_dim=128,
#                               vocab_size=my_dataset.vocab_size, num_classes=3, n_filters=128, kernel_size=4,
#                               pool_size=8, n_lstm=n_lstm[i], dropout=0.2, recurrent_dropout=0.2)
#     my_hybrid.model.summary()
#
#     my_hybrid.train(epoch=2, batch_size=32)
#     val_loss = my_hybrid.get_val_loss()
#     losses[i] = val_loss
#
# print(losses)
# plt.plot(n_lstm, losses)
# plt.grid(True)
# plt.xlabel("Number of LSTM Units")
# plt.ylabel("Validation Loss")
# plt.title("Validation Loss vs. Number of LSTM Units")
# plt.show()

#Epoch
# epoch = [1,2,3,4,5,6,7,8,9,10]
# losses = np.zeros(len(epoch))
#
# for i in range(len(epoch)):
#     my_hybrid = cnn_with_lstm(my_dataset.X_train, my_dataset.y_train, my_dataset.X_val, my_dataset.y_val,
#                               my_dataset.X_test, my_dataset.y_test, my_dataset.max_length, embedding_dim=128,
#                               vocab_size=my_dataset.vocab_size, num_classes=3, n_filters=32, kernel_size=8,
#                               pool_size=6, n_lstm=16, dropout=0.2, recurrent_dropout=0.2)
#     my_hybrid.model.summary()
#
#     my_hybrid.train(epoch=epoch[i], batch_size=32)
#     val_loss = my_hybrid.get_val_loss()
#     losses[i] = val_loss
#
# print(losses)
# plt.plot(epoch, losses)
# plt.grid(True)
# plt.xlabel("Number of Epochs")
# plt.ylabel("Validation Loss")
# plt.title("Validation Loss vs. Number of Epochs Used to Train the Model")
# plt.show()

# #Batch size
# batchsize = [8, 16, 32, 64, 128, 256]
# losses = np.zeros(len(batchsize))
#
# for i in range(len(batchsize)):
#     my_hybrid = cnn_with_lstm(my_dataset.X_train, my_dataset.y_train, my_dataset.X_val, my_dataset.y_val,
#                               my_dataset.X_test, my_dataset.y_test, my_dataset.max_length, embedding_dim=128,
#                               vocab_size=my_dataset.vocab_size, num_classes=3, n_filters=32, kernel_size=8,
#                               pool_size=6, n_lstm=16, dropout=0.2, recurrent_dropout=0.2)
#     my_hybrid.model.summary()
#
#     my_hybrid.train(epoch=1, batch_size=batchsize[i])
#     val_loss = my_hybrid.get_val_loss()
#     losses[i] = val_loss
#
# print(losses)
# plt.plot(batchsize, losses)
# plt.grid(True)
# plt.xlabel("Batchsize")
# plt.ylabel("Validation Loss")
# plt.title("Validation Loss vs. Batchsize Used to Train Model")
# plt.show()

#Recurrent dropout
# dropout = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
# losses = np.zeros(len(dropout))
#
# for i in range(len(dropout)):
#     my_hybrid = cnn_with_lstm(my_dataset.X_train, my_dataset.y_train, my_dataset.X_val, my_dataset.y_val,
#                               my_dataset.X_test, my_dataset.y_test, my_dataset.max_length, embedding_dim=128,
#                               vocab_size=my_dataset.vocab_size, num_classes=3, n_filters=32, kernel_size=8,
#                               pool_size=6, n_lstm=16, dropout=0.3, recurrent_dropout=dropout[i])
#     my_hybrid.model.summary()
#
#     my_hybrid.train(epoch=1, batch_size=16)
#     val_loss = my_hybrid.get_val_loss()
#     losses[i] = val_loss

# print(losses)
# plt.plot(dropout, losses)
# plt.grid(True)
# plt.xlabel("Dropout")
# plt.ylabel("Validation Loss")
# plt.title("Validation Loss vs. Dropout Used to Train Model")
# plt.show()

my_hybrid = cnn_with_lstm(my_dataset.X_train, my_dataset.y_train, my_dataset.X_val, my_dataset.y_val,
                          my_dataset.X_test, my_dataset.y_test, my_dataset.max_length, embedding_dim=128,
                          vocab_size=my_dataset.vocab_size, num_classes=3, n_filters=128, kernel_size=4,
                          pool_size=8, n_lstm=64, dropout=0.2, recurrent_dropout=0.2)
my_hybrid.model.summary()

my_hybrid.train(epoch=5, batch_size=64)
my_hybrid.test()
print("====================")
print("CNN-LSTM Hybrid F1:")
print(my_hybrid.f1)
print("CNN-LSTM Hybdrid Accuracy:")
print(my_hybrid.accuracy)

#Embedding dimension

# my_lstm_with_attention = lstm_with_attention(my_dataset.X_train, my_dataset.y_train, my_dataset.X_val, my_dataset.y_val, my_dataset.X_test, my_dataset.y_test, my_dataset.max_length, my_dataset.vocab_size, embedding_dim = 128, lstm_units = 32, dense_units = 16, num_classes = 3, dropout = 0.2)
# my_lstm_with_attention.train(epoch = 5, batch_size = 64)
# my_lstm_with_attention.test(tokenizer = my_dataset.tokenizer)



