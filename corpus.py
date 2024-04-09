#====================|Importing Dependencies|====================
import random
import numpy as np
import matplotlib.pyplot as plt

#Keras
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#nltk
from nltk.wsd import lesk

#sklearn
from sklearn.model_selection import train_test_split

#user-defined classes
from tweet import tweet

#====================|Class corpus|====================

#Class for storing information about the dataset
class corpus():
    def __init__(self, dataset, labels):
        self.number_of_tweets = len(labels)

        #raw_dataset contains the raw dataset obtained from the csv files before pre-processing is applied.
        self.raw_dataset = dataset

        #Number of unique words encountered in the training set. Initially set to 0 because no words has been identified yet.
        self.vocab_size = 0

        #Longest tweet encountered in the training set. Initially set to 0 because no tweets' lengths have been measured yet
        self.longest_train_tweet = int(0)
        self.longest_test_tweet = int(0)

        #All tweets in the corpus will have the same length, max_length, so that they can be inputted into the CNN layer. Initially set to 0.
        self.max_length = int(0)

        #clean_dataset contains the preprocessed dataset
        [self.clean_dataset, self.labels] = self.get_clean_dataset(self.raw_dataset, labels)

        #Creates a tokenizer by using keras' tokenizer
        self.tokenizer = Tokenizer()

        #Training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test, self.y_train_scalar_label = self.get_train_test_data()

        #Array for storing the distribution of tweet lengths
        self.X_train_length_distribution
        self.X_test_length_distribution

        #Array for storing the class distribution in the corpus
        self.train_class_distribution = self.get_class_distribution(self.y_train_scalar_label)
        self.test_class_distribution = self.get_class_distribution(self.y_test)

    #Function for getting preprocessed (clean) dataset
    def get_clean_dataset(self, raw_dataset, labels):
        #Lists for storing cleaned dataset and label
        temp_dataset = []
        temp_label = []

        #Loops through all tweets in the dataset
        for i in range(self.number_of_tweets):
            #Creates an object temp_tweet of type tweet for the current tweet in the dataset
            temp_tweet = tweet(raw_dataset[i], labels[i])

            #Uses the get_clean_text() function in tweet to obtain the pre-processed version of the current tweet, and appends it to the temp_dataset list
            temp_dataset.append(temp_tweet.get_clean_text())

            #Appends temp_tweet's label to temp_label
            temp_label.append(int(temp_tweet.get_label()))

        #Returns the cleaned dataset and its corresponding labels
        return [temp_dataset, temp_label]


    #Function for returning the clean tweet, that has index tweet_index
    def get_tweet_text(self, tweet_index):
        return self.clean_dataset[tweet_index]

    #Function for returning the label of the tweet stored in index tweet_index
    def get_tweet_label(self, tweet_index):
        return self.labels[tweet_index]

    #Function for obtaining the dataset's class distribution
    def get_class_distribution(self, input_dataset_label):
        #class_distribution stores the number of tweets belonging to each class. E.g., 0th element of class_distribution = number of class 0 (hate speech) tweets
        class_distribution = [0, 0, 0]

        #Increments one to class_distribution[class_label] if the i^th tweet belongs to class class_label
        for i in range(len(input_dataset_label)):
            class_distribution[input_dataset_label[i]] = class_distribution[input_dataset_label[i]] + 1

        #Returns input dataset's class distribution
        return class_distribution

    #Function for showing class distribution as a pie chart
    def show_class_distribution(self):

        fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (10,10))

        axes[0].pie(self.train_class_distribution, autopct='%1.1f%%', textprops={'fontsize': 14})
        axes[0].set_title("Training Set", fontsize= 16)

        axes[1].pie(self.test_class_distribution, autopct='%1.1f%%', textprops={'fontsize': 14})
        axes[1].set_title("Test Set", fontsize= 16)

        plt.legend(['Hate Speech (0)', 'Offensive (1)', 'Neither (2)'], bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize = 14)
        plt.tight_layout()
        plt.show()

    #Function for sorting dataset
    def sort_dataset(self):

        #X0 stores class 0 tweets, y0 stores class 0 labels
        X0 = []
        y0 = []

        #X1 stores class 1 tweets, y1 stores class 1 labels
        X1 = []
        y1 = []

        #X2 stores class 2 tweets, y2 stores class 2 labels
        X2 = []
        y2 = []

        #Loops through all tweets in the datasets, and appends them to the appropriate lists
        for i in range(self.number_of_tweets):
            #E.g., if the current tweet's label == 0, its content is appended to X0, and its label is appended to y0
            if(self.labels[i] == 0):
                X0.append(self.clean_dataset[i])
                y0.append(self.labels[i])

            elif(self.labels[i] == 1):
                X1.append(self.clean_dataset[i])
                y1.append(self.labels[i])

            elif(self.labels[i] == 2):
                X2.append(self.clean_dataset[i])
                y2.append(self.labels[i])

        #Returns the lists containing the sorted tweets and their labels
        return X0, X1, X2, y0, y1, y2

    #Function for splitting the dataset into training and testing sets
    def get_train_test_data(self):

        #Splits dataset into train and test sets, where 90% of the dataset = training set, and 10% = testing set
        X_train, X_test, y_train, y_test = train_test_split(self.clean_dataset, self.labels, test_size = 0.1)

        #Fits the tokenizer to the training set
        self.tokenizer.fit_on_texts(X_train)

        #Obtaining vocab size
        self.vocab_size = len(self.tokenizer.word_index) + 1

        #Converting words to sequences based on the tokenizer self.tokenizer
        X_train = self.tokenizer.texts_to_sequences(X_train)
        X_test = self.tokenizer.texts_to_sequences(X_test)

        #Finding the longest tweet in the training set
        for i in range(len(X_train)):

            #tweet_length stores the length of the current tweet
            tweet_length = len(X_train[i])

            #If tweet_length is larger than the longest_tweet encountered so far, it is set as the new longest_tweet
            if tweet_length > self.longest_train_tweet:
                self.longest_train_tweet = tweet_length


        for i in range(len(X_test)):
            tweet_length = len(X_test[i])

            if tweet_length > self.longest_test_tweet:
                self.longest_test_tweet = tweet_length

        self.X_train_length_distribution = self.get_tweet_length_distribution(X_train, self.longest_train_tweet)
        self.X_test_length_distribution = self.get_tweet_length_distribution(X_test, self.longest_test_tweet)

        #max_length set to 20 to limit the tweet size to 20
        self.max_length = 20 #remove later

        #Padding sequences to ensure they have the same length (i.e., max_length)
        X_train = pad_sequences(X_train, maxlen = self.max_length)
        X_test = pad_sequences(X_test, maxlen = self.max_length)

        X_train = X_train.astype(np.float32)

        #Convert y_train to one-hot encoding because the models output a 3-unit long array of probabilities of a tweet belong to classes 0, 1, and 2
        y_train_one_hot = to_categorical(y_train, num_classes = 3)
        y_train_one_hot = np.array(y_train_one_hot).astype(np.int32)

        #Keeps y_test as a scalar label (e.g., y_test[i] = 0 if i^th tweet belongs to class 0)
        y_test = np.array(y_test).astype(np.int32)

        #Returns the train and test sets
        return [X_train, X_test, y_train_one_hot, y_test, y_train]

    #Function for getting the vocab size
    def get_vocab_size(self):
        return self.vocab_size

    #Function for obtaining the tweet length distribution
    def get_tweet_length_distribution(self, input_dataset, longest_tweet):

        #The i^th element in return_distribution = number of tweets that are i words long. E.g., if the 10^th element = 20, there are 20 tweets that are 10 words long
        return_distribution = np.zeros(longest_tweet+1)

        #Checks each tweet's length in the input_dataset
        for i in range(len(input_dataset)):
            tweet_length = len(input_dataset[i])
            return_distribution[tweet_length] = return_distribution[tweet_length] + 1

        print(return_distribution)
        return return_distribution

    #second input argument = set type (either test or train)
    def show_tweet_length_distribution(self):

        train_tweet_lengths = np.array([i for i in range(1, len(self.X_train_length_distribution) + 1)])
        test_tweet_lengths = np.array([i for i in range(1, len(self.X_test_length_distribution) + 1)])

        train_length_freqs = self.X_train_length_distribution
        test_length_freqs = self.X_test_length_distribution

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

        axes[0].bar(train_tweet_lengths, train_length_freqs)
        axes[0].set_yscale('log')
        axes[0].set_xlabel("Tweet Length", fontsize=14)
        axes[0].set_ylabel("Frequency", fontsize=14)
        axes[0].set_title("Training Set", fontsize=16)

        axes[1].bar(test_tweet_lengths, test_length_freqs)
        axes[1].set_yscale('log')
        axes[1].set_xlabel("Tweet Length", fontsize=14)
        axes[1].set_ylabel("Frequency", fontsize=14)
        axes[1].set_title("Test Set", fontsize=16)

        plt.tight_layout()
        plt.show()