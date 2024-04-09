#Importing dependencies
import numpy as np
import re
import tensorflow as tf
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import seaborn as sns

class tweet():
    def __init__(self, raw_text, label):
        #raw_text stores the original tweet obtained from the dataset
        self.raw_text = raw_text

        #clean_text stores the pre-processed tweet
        self.clean_text = self.preprocess_text(self.raw_text)

        #label stores the tweet's label
        self.label = label

    #Function for removing HTML entities
    def remove_html_entity(self, input_text):
        #Tweets have hmtl entities (i.e., special characters are represented using &(character name), e.g., &amp = & in the tweet.
        #These symbols all start with &, so the regex that will be used in this function will identify words that start with &, and are followed by characters that are not white spaces, or semicolons.
        regex = r"&[^\s;]+;"

        #Replaces all words identified by the regex with ""
        return_text = re.sub(regex, "", input_text)

        #Returns the text without html entities
        return return_text

    #Function for removing user tags
    def replace_user_tag(self, input_text):
        #Tweets have user tags (e.g., @user_name_123)
        #These tags always start with @, so the regex will identify words that start with @
        regex = r"@([^ ]+)"

        #Replaces all user tags with the word user
        return_text = re.sub(regex, "user", input_text)

        #Returns the text without user tags
        return return_text

    #Function for removing urls
    def remove_url(self, input_text):
        #Some tweets have url links. These usually start with either https:// or www.
        #regex identifies strings that start with https:// or www.
        regex = r"https?://\S+|www\.\S+"

        #Replaces the url link with ""
        return_text = re.sub(regex, "", input_text)

        #Returns the text without the url links
        return return_text

    #Function for removing symbols
    def remove_symbols(self, input_text):
        #List of symbols to be removed
        symbols = ['""', "'", "!", ".", "`", ",", "#", ":", "*", "rt"]

        #Checks if the characters in the input text match with any of the symbols mentioned above
        for i in range(len(symbols)):
            #If one of the symbols in the list symbols (above) is in the tweet, replace it with ''
            input_text = input_text.replace(symbols[i], '')

        #Returns the text without the symbols
        return input_text

    #Function for removing stopwords
    def remove_stopwords(self, input_text):
        #Uses nltk's word tokenizer to tokenize the input_text
        tokenised_text = nltk.word_tokenize(input_text)

        #Creates a set of english stop words
        stopwords_english = set(stopwords.words('english'))

        #List for storing return text (with the stopwords removed)
        return_text = []

        #Compares each word in the input_text with stopwords_english to check if it is a stopword
        for i in range(len(tokenised_text)):
            #Converts all characters in the current word to lowercase
            temp_word = tokenised_text[i].lower()

            #Appends the word to return_text if it is not a stopword
            if(temp_word not in stopwords_english):
                return_text.append(temp_word)

        #Returns text (with the stopwords removed)
        return return_text

    #Function for pre-processing the raw tweet
    def preprocess_text(self, raw_text):
        #Applies html entity, user tag, url, symbol, and stopword removal to the raw text
        temp_text = self.remove_html_entity(raw_text)
        temp_text = self.replace_user_tag(temp_text)
        temp_text = self.remove_url(temp_text)
        temp_text = self.remove_symbols(temp_text)
        temp_text = self.remove_stopwords(temp_text)

        #Returns preprocessed text in the form of a list of words
        return temp_text

    #Function for returning the tweet's clean_text
    def get_clean_text(self):
        return self.clean_text

    #Function for returning the tweet's label
    def get_label(self):
        return self.label