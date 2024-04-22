#====================|Importing Dependencies|====================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#keras
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras import layers

#sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

#user-defined classes
from tweet import tweet
from corpus import corpus

#====================|Class lstm|====================
#class for storing information about the lstm model
class lstm():
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, num_classes, tweet_length, n_lstm, dropout, recurrent_dropout, X_test_text):
        #Train and test sets
        self.X_train = np.reshape(X_train, (X_train.shape[0], tweet_length, -1))
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.X_test_text = X_test_text

        #Number of classes in the classification task
        self.num_classes = num_classes

        #Hyperparameters
        self.n_lstm = n_lstm
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout

        #Model stores architecture information about the lstm model
        self.model = self.create_model()

        #Model Performance
        self.accuracy = 0
        self.f1 = 0

    #Function for creating the lstm model
    def create_model(self):
        model = Sequential()

        #Adds LSTM layer with 64 units, 0.2 dropout rate (20% of input units will be randomly set to 0 during training), 0.1 recurrent dropout (10% of connections between recurrent states set to 0 during training)
        model.add(LSTM(units = self.n_lstm, dropout = self.dropout, recurrent_dropout = self.recurrent_dropout))

        #Adds dropout layer
        model.add(Dropout(0.2))

        #Adds dense layer with relu activation
        model.add(Dense(128, activation = "relu"))

        #Adds another dropout layer
        model.add(Dropout(0.2))

        #Adds dense layer with relu activation
        model.add(Dense(128, activation="relu"))

        #Adds another dropout layer
        model.add(Dropout(0.2))

        #Output layer
        model.add(Dense(3, activation = "softmax"))

        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        return model

    #Function for training the model
    def train(self, epoch, batch_size):
        history = self.model.fit(self.X_train, self.y_train, validation_data = (self.X_val, self.y_val), epochs = epoch, batch_size = batch_size)

        train_loss = history.history['loss']
        val_loss = history.history['val_loss']

        epoch_number = [i for i in range(1, epoch + 1)]

        plt.plot(epoch_number, train_loss, label = 'Training Loss')
        plt.plot(epoch_number, val_loss, label = 'Validation Loss')
        plt.xlabel('Loss vs. Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title("Loss vs. Epoch for LSTM Model")
        plt.grid(True)
        plt.show()

    #Function for getting model summary
    def get_summary(self):
        self.model.summary()

    #Function for testing the model
    def test(self, tokenizer):
        #y_pred_one_hot stores the 3 unit vector outputted by the model.
        y_pred_one_hot = self.model.predict(self.X_test)

        #y_pred will be used to store the predicted label. The model outputs 3 probabilities for the tweet belonging to each class, so y_pred will store the number of the class with the highest probability.
        y_pred = []

        #Converts the 3 unit vector y_pred_one_hot into scalar labels y_pred. Because the dataset's y_test is a scalar. So if we want to compare them, y_pred must also be scalar.
        for i in range(len(self.y_test)):
            #Sets current_y_pred equal to the current vector (y_pred_one_hot[i]) we want to convert to y_pred
            current_y_pred = y_pred_one_hot[i]

            #largest_prob stores the largest probability (of the tweet belonging to a class) encountered so far.
            largest_prob = 0

            #most_likely_class stores the class number that has the highest probability, largest_prob
            most_likely_class = 0

            #Each current_y_pred contains 3 probabilities of the corresponding tweet belonging to classes 0 (probability stored in the first unit), 1 (probability stored in the second unit), or 2 (probability stored in the third unit)
            #This for loop finds the unit containing the largest probability, which is the class the tweet most likely belongs to, and sets that as the most_likely_class
            for j in range(self.num_classes):
                if(current_y_pred[j] > largest_prob):
                    largest_prob = current_y_pred[j]
                    most_likely_class = j

            #Appends the most_likely_class to the list y_pred
            y_pred.append(int(most_likely_class))

        y_pred = np.array(y_pred)

        #Building the confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', annot_kws={"size": 10})  # Adjust the size of the annotation text if needed
        plt.xlabel('Predicted Value')
        plt.ylabel('True Value')
        plt.suptitle("Confusion Matrix for LSTM Model")
        plt.gca().set_aspect('auto')  # Set aspect ratio to auto
        plt.show()

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