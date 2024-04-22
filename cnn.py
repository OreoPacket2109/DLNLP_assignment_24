#====================|Importing Dependencies|====================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Keras
from keras import Sequential
from tensorflow.keras import layers
from keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten

#Sklearn
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import f1_score

#====================|Class cnn|====================

#Class for storing details about the cnn, and for training and testing the cnn
class cnn():
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, num_classes, max_length, n_filter, kernel_size, pool_size, X_test_text):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.X_test_text = X_test_text

        #Hyperparameters
        self.kernel_size = kernel_size
        self.n_filter = n_filter
        self.pool_size = pool_size

        #Length of each
        self.max_length = max_length
        self.num_classes = num_classes
        self.model = self.create_model()

        #Model performance
        self.accuracy = 0
        self.f1 = 0

    #Function for training the model
    def train(self, epoch, batch_size):
        history = self.model.fit(self.X_train, self.y_train, validation_data = (self.X_val, self.y_val), epochs = epoch, batch_size = batch_size)

        train_loss = history.history['loss']
        val_loss = history.history['val_loss']

        epoch_number = [i for i in range(1, epoch + 1)]

        plt.plot(epoch_number, train_loss, label = 'Training Loss')
        plt.plot(epoch_number, val_loss, label = 'Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title("Loss vs. Epoch for CNN Model")
        plt.grid(True)
        plt.show()

    #Function for creating the model
    def create_model(self):
        model = Sequential()

        #Reshapes the input
        model.add(layers.Reshape((self.max_length, 1), input_shape = (self.max_length,)))

        #Adds convolution layer
        model.add(layers.Conv1D(filters = self.n_filter, kernel_size = self.kernel_size, activation = 'relu'))

        #Adds max pooling layer
        model.add(layers.MaxPooling1D(self.pool_size))

        #Adds dropout layer, with 10% dropout
        model.add(Dropout(0.2))

        #Adds dense layer
        model.add(layers.Dense(self.n_filter, activation = 'relu'))

        #Adds global max pooling layer
        model.add(layers.GlobalMaxPooling1D())

        #Adds output layer
        model.add(layers.Dense(self.num_classes, activation = 'softmax'))

        #Compiles model
        model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

        return model

    #Function for returning model summary
    def get_summary(self):
        self.model.summary()

    #Functon for testing the model
    def test(self):
        #y_pred_one_hot stores the 3 unit vector output from the model
        y_pred_one_hot = self.model.predict(self.X_test)

        #y_pred will be used to store the predicted class for each test tweet -- this will be a scalar label (either 0, 1, or 2) and not a vector like y_pred_one_hot
        y_pred = []

        #Sets y_pred to be the class that has the highest probability in each tweet's y_pred_one_hot
        for i in range(len(self.y_test)):
            #current_y_pred stores the current prediction that is being considered
            current_y_pred = y_pred_one_hot[i]

            #largest_prob stores the largest prediction encountered so far in current_y_pred
            largest_prob = 0

            #most_likely_class is the class that has the largest probability in current_y_pred
            most_likely_class = 0

            #for loop loops through all self.num_classes (=3 classes for this dataset) to find the class that has the highest probability
            for j in range(self.num_classes):
                if(current_y_pred[j] > largest_prob):
                    largest_prob = current_y_pred[j]
                    most_likely_class = j

            #Appends the current prediction to y_pred
            y_pred.append(int(most_likely_class))

        y_pred = np.array(y_pred)

        #Plots confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', annot_kws={"size": 10})  # Adjust the size of the annotation text if needed
        plt.xlabel('Predicted Value')
        plt.ylabel('True Value')
        plt.suptitle("Confusion Matrix for CNN Model")
        plt.gca().set_aspect('auto')  # Set aspect ratio to auto
        plt.show()

        #Prints out model test accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average=None)

        self.accuracy = accuracy
        self.f1 = f1
        self.get_mispredicted_tweet(y_pred, 50)

    def get_mispredicted_tweet(self, y_pred, n_tweets):
        for i in range(n_tweets):
            if(y_pred[i]!=self.y_test[i]):
                if (int(y_pred[i]) == 2):
                    print("Predicted class: " + str(y_pred[i]) + "; True class: " + str(self.y_test[i]) + "; Sentence: " + str(self.X_test_text[i]))