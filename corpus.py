#====================|Importing Dependencies|====================
import numpy as np
import matplotlib.pyplot as plt

#Keras
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
        self.longest_val_tweet = int(0)
        self.longest_test_tweet = int(0)

        #All tweets in the corpus will have the same length, max_length, so that they can be inputted into the CNN layer. Initially set to 0.
        self.max_length = int(0)

        #clean_dataset contains the preprocessed dataset
        self.class_distribution = [0, 0, 0]
        self.smallest_class_population = 10000000
        [self.clean_dataset, self.labels] = self.get_clean_dataset(self.raw_dataset, labels)
        [self.balanced_dataset, self.balanced_labels] = self.balance_dataset(self.clean_dataset, self.labels)

        #Creates a tokenizer by using keras' tokenizer
        self.tokenizer = Tokenizer()

        #Training and testing sets
        self.X_test_text = []
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test, self.y_train_scalar_label, self.y_val_scalar_label = self.get_train_val_test_data()

        #Array for storing the distribution of tweet lengths
        self.X_train_length_distribution
        self.X_val_length_distribution
        self.X_test_length_distribution

        #Array for storing the class distribution in the corpus
        self.train_class_distribution = self.get_class_distribution(self.y_train_scalar_label)
        self.val_class_distribution = self.get_class_distribution(self.y_val_scalar_label)
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

            self.class_distribution[int(labels[i])] = self.class_distribution[int(labels[i])] + 1

        for i in range(3):
            if (self.class_distribution[i] < self.smallest_class_population):
                self.smallest_class_population = self.class_distribution[i]

        print(self.smallest_class_population)
        #Returns the cleaned dataset and its corresponding labels
        return [temp_dataset, temp_label]

    def show_old_distribution(self):
        old_distribution = [0, 0, 0]
        for i in range(self.number_of_tweets):
            old_distribution[int(self.labels[i])] = old_distribution[int(self.labels[i])] + 1
        plt.figure(figsize=(12, 6))
        plt.pie(old_distribution, labels = ["Hate Speech (0)", "Offensive (1)", "Neither (2)"], autopct='%1.1f%%', textprops={'fontsize': 18})
        plt.title("Original Dataset's Class Distribution", fontsize = 22)
        plt.legend(['Hate Speech (0)', 'Offensive (1)', 'Neither (2)'], bbox_to_anchor=(1.05, 1.0), loc='upper left',
                   fontsize=18)
        plt.tight_layout()
        plt.show()

    def balance_dataset(self, dataset, labels):
        current_distribution = [0, 0, 0]
        return_dataset = []
        return_label = []

        for i in range(3):
            for j in range(self.number_of_tweets):
                if(int(labels[j]) == i):
                    return_dataset.append(dataset[j])
                    return_label.append(labels[j])
                    current_distribution[i] = current_distribution[i] + 1

                if(current_distribution[i] >= self.smallest_class_population):
                    break

        print(current_distribution)
        return return_dataset, return_label


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

        #Creates a 1x3 plot. First subplot shows the training set's class distribution. Second subplot shows the val set's class distribution. Third subplot shows the test set's class distribution.
        fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (10,10))

        #Creating first subplot
        axes[0].pie(self.train_class_distribution, autopct='%1.1f%%', textprops={'fontsize': 16})
        axes[0].set_title("Training Set", fontsize= 18)

        #Creating second subplot
        axes[1].pie(self.val_class_distribution, autopct='%1.1f%%', textprops={'fontsize': 16})
        axes[1].set_title("Validation Set", fontsize= 18)

        #Creating second subplot
        axes[2].pie(self.test_class_distribution, autopct='%1.1f%%', textprops={'fontsize': 16})
        axes[2].set_title("Test Set", fontsize= 18)

        #Adding legend to the plot
        plt.legend(['Hate Speech (0)', 'Offensive (1)', 'Neither (2)'], bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize = 16)
        plt.tight_layout()
        plt.show()

    #Function for splitting the dataset into training and testing sets
    def get_train_val_test_data(self):

        #Splits dataset into train and test sets, where 80% of the dataset = training set, and 20% = val set
        X_train, X_val, y_train, y_val = train_test_split(self.balanced_dataset, self.balanced_labels, test_size = 0.2)

        #Splits the train set into train and validation
        X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size = 0.5)

        #Fits the tokenizer to the training set
        self.tokenizer.fit_on_texts(X_train)

        #Obtaining vocab size
        self.vocab_size = len(self.tokenizer.word_index) + 1

        #Storing X_test in text form so we can analyse which sentence the model misclassifies
        self.X_test_text = X_test

        #Converting words to sequences based on the tokenizer self.tokenizer
        X_train = self.tokenizer.texts_to_sequences(X_train)
        X_val = self.tokenizer.texts_to_sequences(X_val)
        X_test = self.tokenizer.texts_to_sequences(X_test)

        #Finding the longest tweet in the training set
        for i in range(len(X_train)):
            #tweet_length stores the length of the current tweet
            tweet_length = len(X_train[i])

            #If tweet_length is larger than the longest_train_tweet encountered so far, it is set as the new longest_train_tweet
            if tweet_length > self.longest_train_tweet:
                self.longest_train_tweet = tweet_length

        #Finding the longest tweet in the validation set
        for i in range(len(X_val)):
            #tweet_length stores the length of the current tweet
            tweet_length = len(X_val[i])

            #If tweet_length is larger than the longest_train_tweet encountered so far, it is set as the new longest_train_tweet
            if tweet_length > self.longest_val_tweet:
                self.longest_val_tweet = tweet_length

        #Finding the longest tweet in the test set
        for i in range(len(X_test)):
            #tweet_length stores the length of the current tweet from the test set
            tweet_length = len(X_test[i])

            #If tweet_length is larger than the longest_test_tweet encountered so far, it is set as the new longest_test_tweet
            if tweet_length > self.longest_test_tweet:
                self.longest_test_tweet = tweet_length

        #Gets the distribution of tweet lengths for the training and test sets
        self.X_train_length_distribution = self.get_tweet_length_distribution(X_train, self.longest_train_tweet)
        self.X_val_length_distribution = self.get_tweet_length_distribution(X_val, self.longest_val_tweet)
        self.X_test_length_distribution = self.get_tweet_length_distribution(X_test, self.longest_test_tweet)

        #max_length set to 20 to limit the tweet size to 20
        self.max_length = 20 #remove later

        #Padding sequences to ensure they have the same length (i.e., max_length)
        X_train = pad_sequences(X_train, maxlen = self.max_length)
        X_val = pad_sequences(X_val, maxlen = self.max_length)
        X_test = pad_sequences(X_test, maxlen = self.max_length)

        X_train = X_train.astype(np.float32)
        X_val = X_val.astype(np.float32)

        #Convert y_train and y_val to one-hot encoding because the models output a 3-unit long array of probabilities of a tweet belong to classes 0, 1, and 2
        y_train_one_hot = to_categorical(y_train, num_classes = 3)
        y_train_one_hot = np.array(y_train_one_hot).astype(np.int32)

        y_val_one_hot = to_categorical(y_val, num_classes = 3)
        y_val_one_hot = np.array(y_val_one_hot).astype(np.int32)

        #Keeps y_test as a scalar label (e.g., y_test[i] = 0 if i^th tweet belongs to class 0)
        y_test = np.array(y_test).astype(np.int32)

        #Returns the train and test sets. y_train_one_hot will be used to fit the model (since the model wants one-hot-encoded labels), while y_train (the scalar version of the labels) will be used to find the train set's class distribution (since get_class_distirbution checks a tweet's class by checking its scalar label (whether it's 0, 1, or 2, not [1 0 0], [0 1 0], or [0 0 1])
        return [X_train, X_val, X_test, y_train_one_hot, y_val_one_hot, y_test, y_train, y_val]

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

        return return_distribution

    #Function for showing the distribution of tweet lengths for the train and test sets
    def show_tweet_length_distribution(self):

        #Creates arrays containing numbers from 0 to the largest tweet in the training/testing set. This will be used as the plot's x-axis
        train_tweet_lengths = np.array([i for i in range(1, len(self.X_train_length_distribution) + 1)])
        val_tweet_lengths = np.array([i for i in range(1, len(self.X_val_length_distribution) + 1)])
        test_tweet_lengths = np.array([i for i in range(1, len(self.X_test_length_distribution) + 1)])

        #Creates array for storing the number of tweets with 0, 1, 2, ... words. E.g., train_length_freqs[10] = number of tweets from the training set that are 10 words long. This will be used as the plot's y-axis.
        train_length_freqs = self.X_train_length_distribution
        val_length_freqs = self.X_val_length_distribution
        test_length_freqs = self.X_test_length_distribution

        #Creates a 1x3 plot. First subplot = training set's tweet length distribution. Second subplot = val set's tweet length distribution. Third subplot = test set's tweet length distribution.
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

        #Plotting the first subplot
        axes[0].bar(train_tweet_lengths, train_length_freqs)
        #axes[0].set_yscale('log')
        axes[0].set_xlabel("Tweet Length", fontsize=16)
        axes[0].set_ylabel("Frequency", fontsize=16)
        axes[0].set_title("Training Set", fontsize=18)

        #Plotting the second subplot
        axes[1].bar(val_tweet_lengths, val_length_freqs)
        #axes[1].set_yscale('log')
        axes[1].set_xlabel("Tweet Length", fontsize=16)
        axes[1].set_ylabel("Frequency", fontsize=16)
        axes[1].set_title("Test Set", fontsize=18)

        #Plotting the third subplot
        axes[2].bar(test_tweet_lengths, test_length_freqs)
        #axes[2].set_yscale('log')
        axes[2].set_xlabel("Tweet Length", fontsize=16)
        axes[2].set_ylabel("Frequency", fontsize=16)
        axes[2].set_title("Test Set", fontsize=18)

        #Showing the plot
        plt.tight_layout()
        plt.show()