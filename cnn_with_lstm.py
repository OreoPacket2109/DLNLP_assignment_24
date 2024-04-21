#====================|Importing Dependencies|====================
import numpy as np
import matplotlib.pyplot as plt

#Keras
import tensorflow as tf
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Embedding
from keras.utils import to_categorical

#Sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import seaborn as sns

#nltk
import nltk
nltk.download('wordnet')

#user-defined classes

#====================|Class cnn_with_lstm|====================

#Contains information about the hybrid model, and functions for training and testing the model
class cnn_with_lstm():
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, max_seq_length, embedding_dim, vocab_size, num_classes, n_filters, kernel_size, pool_size, n_lstm, dropout, recurrent_dropout):
        #Train and test sets
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

        #Number of classes in the dataset
        self.num_classes = num_classes

        #Length of each tweet (sequence)
        self.max_seq_length = max_seq_length

        #Embedding dimension
        self.embedding_dim = embedding_dim

        #CNN-LSTM model hyperparameters
        self.n_filters = n_filters #number of filters in the convolution layer
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.n_lstm = n_lstm #number of lstm units
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout

        #Number of words in the vocabulary (based on the training set)
        self.vocab_size = vocab_size

        #Model object
        self.model = self.create_model()

        #Model performance
        self.accuracy = 0
        self.f1 = 0

    def create_model(self):
        model = Sequential()

        #Embedding layer
        model.add(Embedding(input_dim = self.vocab_size, output_dim = self.embedding_dim, input_length = self.max_seq_length))

        #CNN layer
        model.add(Conv1D(filters = self.n_filters, kernel_size = self.kernel_size, activation = 'relu', input_shape = (self.max_seq_length, self.embedding_dim)))
        model.add(MaxPooling1D(pool_size = self.pool_size))

        #LSTM layer
        model.add(LSTM(units = self.n_lstm, dropout = self.dropout, recurrent_dropout = self.recurrent_dropout, return_sequences = True))

        #Flatten layer
        model.add(Flatten())

        #Output layer with softmax activation function
        model.add(Dense(self.num_classes, activation = 'softmax'))

        #Compiling the model
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

        return model

    #Function for training the model
    def train(self, epoch, batch_size):
        #Stores the model's performance at each epoch
        history = self.model.fit(self.X_train, self.y_train, validation_data = (self.X_val, self.y_val), epochs = epoch, batch_size = batch_size)

        #Stores the model's training and validation accuracy, to be used for the plot later
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']

        #Array for storing the epoch number, to be used for the plot's x-axis
        epoch_number = [i for i in range(1, epoch + 1)]

        #Plotting the accuracy vs. epoch curve
        plt.plot(epoch_number, train_loss, label = 'Training Loss')
        plt.plot(epoch_number, val_loss, label = 'Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss vs. Epoch')
        plt.legend()
        plt.grid(True)
        plt.show()

    #Function for testing the model
    def test(self):
        #y_pred_one_hot stores the one-hot encoded ouput the model
        y_pred_one_hot = self.model.predict(self.X_test)

        #y_pred stores the most likely label for each tweet in X_test. This corresponds to the class with the largest probability in the tweet's y_pred_one_hot
        y_pred = []

        #Finds the class that has the highest probability for each tweet, and sets it as the tweet's y_pred
        for i in range(len(self.y_test)):
            current_y_pred = y_pred_one_hot[i]
            largest_prob = 0
            most_likely_class = 0

            for j in range(self.num_classes):
                if(current_y_pred[j] > largest_prob):
                    largest_prob = current_y_pred[j]
                    most_likely_class = j

            #Appends the i^th tweet's y_pred to the list y_pred
            y_pred.append(int(most_likely_class))

        #Converts y_pred into an array
        y_pred = np.array(y_pred)

        #Shows confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', annot_kws={"size": 10})  # Adjust the size of the annotation text if needed
        plt.xlabel('Predicted Value')
        plt.ylabel('True Value')
        plt.suptitle("Confusion Matrix for CNN-LSTM Hybrid Model")
        plt.gca().set_aspect('auto')  # Set aspect ratio to auto
        plt.show()

        y_test_one_hot = to_categorical(self.y_test, num_classes = 3)
        y_test_one_hot = np.array(y_test_one_hot).astype(np.int32)

        #Getting accuracy and f1 score
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average=None)
        self.accuracy = accuracy
        self.f1 = f1

    def get_val_loss(self):
        val_loss = self.model.evaluate(self.X_val, self.y_val)
        print(val_loss[0])
        return val_loss[0]