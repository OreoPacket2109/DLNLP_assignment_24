from keras import Input, Model
from keras.layers import Embedding, LSTM, Attention, Dense
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import seaborn as sns
import numpy as np


class lstm_with_attention():
    def __init__(self, X_train, y_train, X_test, y_test, input_length, vocab_size, embedding_dim, lstm_units, dense_units, num_classes, dropout):
        #Train and test sets
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test #scalar labels

        #Length of each sequence, which is equal to my_dataset.max_length
        self.input_length = input_length

        #Number of words in the vocabulary
        self.vocab_size = vocab_size

        #Tunable hyperparameters
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout = dropout

        #Number of classes in the dataset
        self.num_classes = num_classes

        #Model
        self.model = self.create_model()

    #Function for creating model
    def create_model(self):
        #Adds input layer
        inputs = Input(shape = (self.input_length,))

        #Adds embedding layer. Input dimension depends on the vocab_size.
        embedding = Embedding(input_dim = self.vocab_size, output_dim = self.embedding_dim)(inputs)

        #Adds LSTM layer
        lstm = LSTM(units = self.lstm_units, dropout = self.dropout)(embedding)

        #Adds attention layer. Output of the lstm layer is used as both the query and value vectors to enable self-attention.
        attention = Attention()([lstm, lstm])

        #Adds dense layer
        dense = Dense(units = self.dense_units, activation = 'relu')(attention)

        #Adds output layer
        outputs = Dense(units = self.num_classes, activation = 'softmax')(dense)

        #Defines model
        model = Model(inputs = inputs, outputs = outputs)

        #Compiles model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def train(self, epoch, batch_size):
        history = self.model.fit(self.X_train, self.y_train, epochs = epoch, batch_size = batch_size)

        plt.plot(history.history['accuracy'])

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
        plt.suptitle("Confusion Matrix")
        plt.gca().set_aspect('auto')  # Set aspect ratio to auto
        plt.show()

        #Printing out the accuracy score
        accuracy = accuracy_score(self.y_test, y_pred)
        print(accuracy)

        #Printing out F1 score
        f1 = f1_score(self.y_test, y_pred, average = 'micro')

        print(f1)





