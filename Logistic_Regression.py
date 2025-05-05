import pandas as pd
import time
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression


'''
-# Loads in the datasets and puts them into a pandas dataframe, clears all "NaN" values

-# Returns the a pandas dataframe with the removed features
'''
def load_dataset(filename, label):
    dataframe = pd.read_csv(filename)
    dataframe.drop(dataframe.columns[[0, 1, 2, 3, 4, 34]], axis=1, inplace=True)

    dataframe['label'] = label

    dataframe.dropna(inplace=True)

    return dataframe

'''
-# This function formats the data from the given csv files, it loads each of the datasets and
-# adds the benign and malicious sets together. It then shuffles and then scales the testing and training
-# data.

-# Can change the shuffle of the data by inputting a random_state variable, defaults to 21

-# Creates the labels for 0 being benign, 1 being Malicious

-# Return: The function returns 4 different formated data:
-#    Training set, Training labels, Testing set, Testing labels
'''
def format_data(benign, benignTest, Malicious, MaliciousTest, random_state=21):
    benignTrainingData = load_dataset(benign, 0)
    maliciousTrainingData = load_dataset(benignTest, 1)
    benignTestingData = load_dataset(Malicious, 0)
    maliciousTestingData = load_dataset(MaliciousTest, 1)

    combinedTrainingData = pd.concat([benignTrainingData, maliciousTrainingData])
    combinedTestingData = pd.concat([benignTestingData, maliciousTestingData])

    X_train = combinedTrainingData.drop(columns=['label'])
    y_train = combinedTrainingData['label']
    X_test = combinedTestingData.drop(columns=['label'])
    y_test = combinedTestingData['label']

    X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train, random_state)
    X_test_shuffled, y_test_shuffled = shuffle(X_test, y_test, random_state)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train_shuffled)
    X_test_scaled = scaler.transform(X_test_shuffled)

    return X_train_scaled, y_train_shuffled, X_test_scaled, y_test_shuffled
    #return X_train_shuffled, y_train_shuffled, X_test_shuffled, y_test_shuffled


def main():
    #Creates the test, train, and label sets
    X_train, y_train, X_test, y_test = format_data("l2-benign_training.csv", "l2-malicious_training.csv", "l2-benign_testing.csv", "l2-malicious_testing.csv")

    log_reg = LogisticRegression(max_iter=10000)
    #start = time.perf_counter()
    log_reg.fit(X_train, y_train)
    #end = time.perf_counter()

    pred = log_reg.predict(X_test)

    #Grabs the metric scores
    acc = accuracy_score(y_test, pred)
    recall = recall_score(y_test, pred)
    precision = precision_score(y_test, pred)
    f1score = f1_score(y_test, pred)

    #Creates the confusion matrix
    confusionMatrix = confusion_matrix(y_test, pred)
    confusionMatrixDF = pd.DataFrame(confusionMatrix, index=["Actual Benign", "Actual Malicious"], columns=["Predicted Benign", "Predicted Malicious"])

    print (f"Accuracy: {acc}")
    print (f"Recall: {recall}")
    print (f"Precision: {precision}")
    print (f"F-1 Score:  {f1score}")
    print("Confusion Matrix" +str(confusionMatrixDF))

main()