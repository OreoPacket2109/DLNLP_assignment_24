#====================|Importing Dependencies|====================
import numpy as np
import matplotlib.pyplot as plt

#Keras
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Embedding

#Sklearn
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

#nltk
import nltk
nltk.download('wordnet')

#user-defined classes

#====================|Class cnn_with_lstm|====================

#Contains information about the hybrid model, and functions for training and testing the model
class cnn_with_lstm():
    def __init__(self, X_train, y_train, X_test, y_test, max_seq_length, embedding_dim, vocab_size, num_classes):
        #Train and test sets
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        #Number of classes in the dataset
        self.num_classes = num_classes

        #Length of each tweet (sequence)
        self.max_seq_length = max_seq_length

        #Embedding dimension
        self.embedding_dim = embedding_dim

        #Number of words in the vocabulary (based on the training set)
        self.vocab_size = vocab_size

        #Model object
        self.model = self.create_model()

    def create_model(self, filters = 64, kernel_size = 3, pool_size = 2, units = 128, dropout = 0.2, recurrent_dropout = 0.2):
        model = Sequential()

        #Embedding layer
        model.add(Embedding(input_dim = self.vocab_size, output_dim = self.embedding_dim, input_length = self.max_seq_length))

        #CNN layer
        model.add(Conv1D(filters = filters, kernel_size = kernel_size, activation = 'relu', input_shape = (self.max_seq_length, self.embedding_dim)))
        model.add(MaxPooling1D(pool_size = pool_size))

        #LSTM layer
        model.add(LSTM(units = units, dropout = dropout, recurrent_dropout = recurrent_dropout, return_sequences = True))

        #Flatten layer
        model.add(Flatten())

        #Output layer with softmax activation function
        model.add(Dense(self.num_classes, activation = 'softmax'))

        #Compiling the model
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

        return model

    #Function for training the model
    def train(self, epoch, batch_size):
        #history stores the model's across each epoch
        history = self.model.fit(self.X_train, self.y_train, epochs = epoch, batch_size = batch_size)

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
        plt.suptitle("Confusion Matrix")
        plt.gca().set_aspect('auto')  # Set aspect ratio to auto
        plt.show()

        #Printing out the accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        print(accuracy)