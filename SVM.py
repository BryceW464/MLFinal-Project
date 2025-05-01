import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

'''
-# Loads in the datasets and puts them into a pandas dataframe, clears all "NaN" values

-# Returns the a pandas dataframe with the removed features
'''
def load_dataset(filename, label):
    dataframe = pd.read_csv(filename)
    dataframe.drop(dataframe.columns[[0, 1, 4, 34]], axis=1, inplace=True)

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

    X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train, random_state=random_state)
    X_test_shuffled, y_test_shuffled = shuffle(X_test, y_test, random_state=random_state)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_shuffled)
    X_test_scaled = scaler.transform(X_test_shuffled)

    return X_train_scaled, y_train_shuffled, X_test_scaled, y_test_shuffled

"""
-# This function takes in data and tries to figure out the best hyperparameters for a
-# SVM with a 'rbf' kernel, can input print_Params=True to see the final picked parameters

-# Returns the hyperparameters "C" and "gamma", it prints out the best hyperparameters for "C" and "Gamma"
"""
def hyper_para(X_data, Y_data, print_Params=False):

    param_grid = {
    'C': [0.1, 1, 10, 100, 1000],  
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
    }

    svm_model = svm.SVC(kernel="rbf", verbose=False)

    # Perform GridSearch with 5-fold cross-validation
    grid_search = GridSearchCV(svm_model, param_grid, refit=True)
    grid_search.fit(X_data, Y_data)

    # Print the best parameters and the best score
    if print_Params == True:
        print("Best Parameters: ", grid_search.best_params_)
        print("Best Cross-validation Score: ", grid_search.best_score_)

    return grid_search.best_params_

'''
-# Creates a SVM model with the given training data and labels
-# Takes in the hyperparameters C and gamma

-# Returns a SVM model fitting to the given data
'''
def create_model(training, labels, C=1, gamma=.01):
    svm_model = svm.SVC(C=C, kernel="rbf", gamma=gamma)

    svm_model.fit(training, labels)

    return svm_model

'''
-# This function takes in a svm model and given test data and labels
-# Using the inputted model and data/labels, creates a confusion matrix as well as prints out
-# accuracy, recall, precision, and the related confusionMatrix for the model to the testing data
'''
def statistics(svm_model, test_data, test_labels):
    y_prediction = svm_model.predict(test_data)

    confusionMatrix = confusion_matrix(test_labels, y_prediction)
    confusionMatrixDF = pd.DataFrame(confusionMatrix, index=["Actual Benign", "Actual Malicious"], columns=["Predicted Benign", "Predicted Malicious"])
    accuracy = accuracy_score(test_labels, y_prediction)
    recall = recall_score(test_labels, y_prediction)
    precision = precision_score(test_labels, y_prediction)

    print(f"accuracy: {accuracy}")
    print(f"recall: {recall}")
    print(f"precision: {precision}\n")
    print("Confusion Matrix" + str(confusionMatrixDF))


def main():
    benignTraining = "l2-benign_training.csv"
    benignTesting = "l2-malicious_training.csv"
    MaliciousTraining = "l2-benign_testing.csv"
    MaliciousTesting = "l2-malicious_testing.csv"

    X_train, y_train, X_test, y_test = format_data(benignTraining, benignTesting, MaliciousTraining, MaliciousTesting, random_state=42)

    params = hyper_para(X_train, y_train, print_Params=True)

    svm_model = create_model(X_train, y_train, C=params['C'], gamma=params['gamma'])

    statistics(svm_model, X_test, y_test)

if __name__ == '__main__':
    main()