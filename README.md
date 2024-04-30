# DLNLP Assignment 2023-24
## Project Description
This project contains the code for the ELEC0141 assignment. The code develops a hybrid CNN-LSTM model for classifying Kaggle's "Hate Speech and Offensive Language Dataset" into three classes: hateful (class 0), offensive (class 1), and neutral (class 2). The hybrid CNN-LSTM model consists of a CNN layer, followed by an LSTM layer. 

For benchmarking purposes, a pure CNN model and a pure LSTM model are also developed in this code, and tested on the same dataset as the hybrid model. Each model's accuracy, F1 score, and confusion matrices are then compared to determine which model had the best performance on the dataset.

## Project Organisation
The project folder is organised as follows:
- DLNLP_assignment_24
  - Datasets
     - dataset.csv
  - cnn.py
  - cnn_with_lstm.py
  - corpus.py
  - lstm.py
  - main.py
  - tweet.py
  - .gitignore
  - README.md
  - requirement.txt

All code related to the assignment must be run from main.py.

cnn.py, cnn_with_lstm.py, and lstm.py are the classes for the CNN, CNN-LSTM, and LSTM models, respectively. They are used to create instances of each model in the program. For instance, cnn.py contains the code for the class cnn(), which is used to create an object of type cnn, which stores information about the CNN model as attributes (such attributes include the number of filters in the CNN model's ocnvolution layer, etc.).

corpus.py is used to store the code for the class corpus, which stores information about the dataset as attributes (e.g., the corpus object has an attribute number_of_tweets which stores the number of tweets in the dataset)

tweet.py is used to store the code for the class tweet, which is used to store information about individual tweets, such as its raw text (i.e., the tweet before it was pre-processed) and its label.

Datasets is a folder which contains the dataset csv file.


## Setting Up the Project
1. Download the project onto the user's local machine.
2. Open the project. The proect folder should look like the one shown in the *Project Organisation* section, except the *Dataset* folder should be empty.
3. Download the dataset. Click [here](https://www.kaggle.com/datasets/thedevastator/hate-speech-and-offensive-language-detection) to open the Kaggle page which contains the dataset.
4. Paste the dataset from Step 3 into the Datasets folder.
5. Download the libraries and packages. Refer to the *Required Packages* section for further details.

## Required Packages
| Package Name | Version |
| -------- | -------- |
| matplotlib | 3.7.2 |
| numpy | 1.24.3 |
| random | 1.2.4 |
| scikit-learn | 1.3.0 |
| tensorflow | 2.15.0 |

For more information, refer to _requirements.txt_ in the repository.

## Running the Project
1. Run the project from main.py
2. The program will train and test all three models on the dataset. The user will get three confusion matrices which show each model's performance on the test set, and each model's accuracy and F1 scores will be printed on the user's terminal.
3. The program will prompt the user to enter a number from 1 to 6. The list below outlines the effect of entering a number from 1 to 6:
    - (1): Prints out the graph for the hybrid model's validation loss vs. kernel size
    - (2): Prints out the graph for the hybrid model's validation loss vs. the number of filters (in the model's CNN layer)
    - (3): Prints out the graph for the hybrid model's validation loss vs. pool size (in the CNN layer's max pooling layer)
    - (4): Prints out the graph for the hybrid model's validation loss vs. the number of lstm units used (in the LSTM layer)
    - (5): Prints out the graph for the hybrid model's validation and training losses vs. the number of epochs used to train the model
    - (6): Terminate the program

## Contributing Guidelines
Contact zceegdu@ucl.ac.uk for permission to edit code.
