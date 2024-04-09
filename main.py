#====================|Importing Dependencies|====================
import numpy as np
import pandas as pd

#User-defined classes
from corpus import corpus
#====================|1. Importing Datasets|====================

#Importing first dataset
dir_a = "Datasets/labeled_data.csv"
df_a = pd.read_csv(dir_a)
dataset_a = df_a['tweet']
labels_a = df_a['class']

#Importing second dataset
dir_b = "Datasets/additional_data.csv"
df_b = pd.read_csv(dir_b)
dataset_b = df_b[df_b['label'] == 'hate']
dataset_b = dataset_b['text']
labels_b = pd.Series(np.zeros(len(dataset_b)))
dataset_b.reset_index(drop = True, inplace = True)

#Combining the two datasets
combined_dataset = pd.concat([dataset_a, dataset_b], ignore_index=True)
labels_b.reset_index(drop = True, inplace = True)
combined_labels = pd.concat([labels_a, labels_b], ignore_index=True)

#====================|2. Dataset Analysis|====================
my_dataset = corpus(combined_dataset, combined_labels)
my_dataset.show_class_distribution()
my_dataset.show_tweet_length_distribution()

