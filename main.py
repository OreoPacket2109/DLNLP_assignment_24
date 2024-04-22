#====================|Importing Dependencies|====================
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

#Keras
from keras.callbacks import History
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Attention
from tensorflow.keras.models import Model
from keras import Sequential, Input
from keras.layers import Embedding, MultiHeadAttention, Concatenate, Reshape

#Sklearn
from sklearn.metrics import confusion_matrix, accuracy_score

#User-defined classes
from corpus import corpus
from lstm import lstm
from cnn import cnn
from cnn_with_lstm import cnn_with_lstm
from lstm_with_attention import lstm_with_attention

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
my_lstm = lstm(my_dataset.X_train, my_dataset.y_train, my_dataset.X_val, my_dataset.y_val, my_dataset.X_test, my_dataset.y_test, 3, my_dataset.max_length, n_lstm = 64, dropout = 0.2, recurrent_dropout = 0.2, X_test_text=my_dataset.X_test_text)
my_lstm.train(epoch = 5, batch_size = 32)
my_lstm.test(my_dataset.tokenizer)
print("====================")
print("LSTM F1:")
print(my_lstm.f1)
print("LSTM Accuracy:")
print(my_lstm.accuracy)

#====================|3.2. CNN|====================
my_cnn = cnn(my_dataset.X_train, my_dataset.y_train, my_dataset.X_val, my_dataset.y_val, my_dataset.X_test, my_dataset.y_test, 3, my_dataset.max_length, n_filter = 128, kernel_size= 4, pool_size= 8, X_test_text=my_dataset.X_test_text)
my_cnn.train(epoch = 5, batch_size = 32)
my_cnn.test()
print("====================")
print('CNN F1:')
print(my_cnn.f1)
print("CNN: Accuracy:")
print(my_cnn.accuracy)
#====================|3.3. Hybrid Model|====================
my_hybrid = cnn_with_lstm(my_dataset.X_train, my_dataset.y_train, my_dataset.X_val, my_dataset.y_val,
                          my_dataset.X_test, my_dataset.y_test, my_dataset.max_length, embedding_dim=128,
                          vocab_size=my_dataset.vocab_size, num_classes=3, n_filters=128, kernel_size=4,
                          pool_size=8, n_lstm=64, dropout=0.2, recurrent_dropout=0.2, X_test_text=my_dataset.X_test_text)
my_hybrid.model.summary()

my_hybrid.train(epoch=5, batch_size=64)
my_hybrid.test()
print("====================")
print("CNN-LSTM Hybrid F1:")
print(my_hybrid.f1)
print("CNN-LSTM Hybdrid Accuracy:")
print(my_hybrid.accuracy)

#====================|4. Hyperparameter Tuning|====================

#User can choose which graph they want to see.
print("The following section prints out the graphs used during the hypertuning process. The user may choose which graph they want to see by inputting the following numbers:")
print("1, to print out the graph for Model Loss vs. Kernel Size")
print("2, to print out the graph for Model Loss vs. Number of Filters")
print("3, to print out the graph for Model Loss vs. Pool Size")
print("4, to print out the graph for Model Loss vs. Number of LSTM units")
print("5, to print out the graph for Model Loss vs. Epoch")
print("6, to terminate program")

user_input = input()

#Keeps looping until user chooses to terminate the program (by entering 6)
while(user_input != "6"):
    #If user enters 1, the program shows them the validation loss vs. kernel size graph
    if user_input == "1":
        # Kernel size
        kernel_sizes = [2, 4, 8, 16]
        losses = np.zeros(len(kernel_sizes))

        #Trains model with different kernel sizes, and records its validation loss
        for i in range(len(kernel_sizes)):
            my_hybrid = cnn_with_lstm(my_dataset.X_train, my_dataset.y_train, my_dataset.X_val, my_dataset.y_val,
                              my_dataset.X_test, my_dataset.y_test, my_dataset.max_length, embedding_dim=128,
                              vocab_size=my_dataset.vocab_size, num_classes=3, n_filters=32, kernel_size=kernel_sizes[i],
                              pool_size=4, n_lstm=32, dropout=0.2, recurrent_dropout=0.2, X_test_text= my_dataset.X_test_text)

            my_hybrid.train(epoch=2, batch_size=32)
            val_loss = my_hybrid.get_val_loss()
            losses[i] = val_loss

        #Plotting the graph
        plt.plot(kernel_sizes, losses)
        plt.grid(True)
        plt.xlabel("Kernel Size")
        plt.ylabel("Validation Loss")
        plt.title("Validation Loss vs Kernel Size")
        plt.show()

    #If user enters 2, the program shows them the validation loss vs. number of conv filters graph
    elif user_input == "2":

        #Number of convoluton filters
        n_filters = [2, 4, 8, 16, 32, 64, 128, 256]
        losses = np.zeros(len(n_filters))

        #Trains the model with different numbers of conv filters, and records their validation loss
        for i in range(len(n_filters)):
            my_hybrid = cnn_with_lstm(my_dataset.X_train, my_dataset.y_train, my_dataset.X_val, my_dataset.y_val,
                              my_dataset.X_test, my_dataset.y_test, my_dataset.max_length, embedding_dim=128,
                              vocab_size=my_dataset.vocab_size, num_classes=3, n_filters=n_filters[i], kernel_size=4,
                              pool_size=4, n_lstm=32, dropout=0.2, recurrent_dropout=0.2, X_test_text= my_dataset.X_test_text)

            my_hybrid.train(epoch=2, batch_size=32)
            val_loss = my_hybrid.get_val_loss()
            losses[i] = val_loss

        #Plots the graph
        plt.plot(n_filters, losses)
        plt.grid(True)
        plt.xlabel("Number of Filters")
        plt.ylabel("Validation Loss")
        plt.title("Validation Loss vs Number of Filters")
        plt.show()

    #If the user chooses 3, shows the graph for validation loss vs. pool size
    elif user_input == "3":
        #Pool size
        pool_size = [2, 3, 4, 5, 7, 8, 10]
        losses = np.zeros(len(pool_size))

        #Trains the model with different max pool sizes, and records its validation loss
        for i in range(len(pool_size)):
            my_hybrid = cnn_with_lstm(my_dataset.X_train, my_dataset.y_train, my_dataset.X_val, my_dataset.y_val,
                              my_dataset.X_test, my_dataset.y_test, my_dataset.max_length, embedding_dim=128,
                              vocab_size=my_dataset.vocab_size, num_classes=3, n_filters=128, kernel_size=4,
                              pool_size=pool_size[i], n_lstm=64, dropout=0.2, recurrent_dropout=0.2, X_test_text= my_dataset.X_test_text)

            my_hybrid.train(epoch=2, batch_size=32)
            val_loss = my_hybrid.get_val_loss()
            losses[i] = val_loss

        #Plots graph
        plt.plot(pool_size, losses)
        plt.grid(True)
        plt.xlabel("Pool Size")
        plt.ylabel("Validation Loss")
        plt.title("Validation Loss vs. Pool Size")
        plt.show()

    #If the user chooses 4, program shows the validation loss vs. number of lstm units graph
    elif user_input == "4":

        #Number of LSTM units
        n_lstm = [2, 4, 8, 16, 32, 64, 128, 256]
        losses = np.zeros(len(n_lstm))

        #Trains the model with different numbers of lstm units, and records the ensuring validation loss
        for i in range(len(n_lstm)):
            my_hybrid = cnn_with_lstm(my_dataset.X_train, my_dataset.y_train, my_dataset.X_val, my_dataset.y_val,
                              my_dataset.X_test, my_dataset.y_test, my_dataset.max_length, embedding_dim=128,
                              vocab_size=my_dataset.vocab_size, num_classes=3, n_filters=128, kernel_size=4,
                              pool_size=8, n_lstm=n_lstm[i], dropout=0.2, recurrent_dropout=0.2, X_test_text= my_dataset.X_test_text)

            my_hybrid.train(epoch=2, batch_size=32)
            val_loss = my_hybrid.get_val_loss()
            losses[i] = val_loss

        #Plots the graph
        plt.plot(n_lstm, losses)
        plt.grid(True)
        plt.xlabel("Number of LSTM Units")
        plt.ylabel("Validation Loss")
        plt.title("Validation Loss vs. Number of LSTM Units")
        plt.show()

    #If the user chooses 5, program plots the validation loss vs. epoch curve
    elif user_input == "5":

        #Epoch
        epoch = [1,2,3,4,5,6,7,8,9,10]
        losses = np.zeros(len(epoch))

        history = History()
        my_hybrid = cnn_with_lstm(my_dataset.X_train, my_dataset.y_train, my_dataset.X_val, my_dataset.y_val,
                                  my_dataset.X_test, my_dataset.y_test, my_dataset.max_length, embedding_dim=128,
                                  vocab_size=my_dataset.vocab_size, num_classes=3, n_filters=32, kernel_size=8,
                                  pool_size=6, n_lstm=16, dropout=0.2, recurrent_dropout=0.2,
                                  X_test_text=my_dataset.X_test_text)

        my_hybrid.train(epoch = 10, batch_size=32)

    #If user enters 6, terminate program
    elif user_input == "6":
        break

    #If user doesn't enter something from 1 to 6, request another input
    else:
        print("Please enter a number from 1 to 6.")

    print("Input:")
    print("1, to print out the graph for Model Loss vs. Kernel Size")
    print("2, to print out the graph for Model Loss vs. Number of Filters")
    print("3, to print out the graph for Model Loss vs. Pool Size")
    print("4, to print out the graph for Model Loss vs. Number of LSTM units")
    print("5, to print out the graph for Model Loss vs. Epoch")
    print("6, to terminate program")

    user_input = input()
