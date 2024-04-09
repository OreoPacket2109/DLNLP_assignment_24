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
        self.raw_text = raw_text
        self.clean_text = self.preprocess_text(self.raw_text)
        self.label = label

    def remove_html_entity(self, input_text):
        regex = r"&[^\s;]+;"
        return_text = re.sub(regex, "", input_text)
        return return_text

    def replace_user_tag(self, input_text):
        regex = r"@([^ ]+)"
        return_text = re.sub(regex, "user", input_text)
        return return_text

    def remove_url(self, input_text):
        regex = r"https?://\S+|www\.\S+"
        return_text = re.sub(regex, "", input_text)
        return return_text

    def remove_symbols(self, input_text):
        symbols = ['""', "'", "!", ".", "`", ",", "#", ":", "*", "rt"]

        for i in range(len(symbols)):
            input_text = input_text.replace(symbols[i], '')

        return input_text

    def remove_stopwords(self, input_text):
        tokenised_text = nltk.word_tokenize(input_text)
        stopwords_english = set(stopwords.words('english'))
        return_text = []

        for i in range(len(tokenised_text)):
            temp_word = tokenised_text[i].lower()

            if(temp_word not in stopwords_english):
                return_text.append(temp_word)

        return return_text

    def preprocess_text(self, raw_text):
        temp_text = self.remove_html_entity(raw_text)
        temp_text = self.replace_user_tag(temp_text)
        temp_text = self.remove_url(temp_text)
        temp_text = self.remove_symbols(temp_text)
        temp_text = self.remove_stopwords(temp_text)
        return temp_text

    def get_clean_text(self):
        return self.clean_text

    def get_label(self):
        return self.label