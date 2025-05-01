import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

def load_dataset(filename, label):
    dataframe = pd.read_csv(filename)
    dataframe.drop(dataframe.columns[[0, 1, 2, 3, 4, 34]], axis=1, inplace=True)

    dataframe['label'] = label

    dataframe.dropna(inplace=True)

    return dataframe

def format_data(benign, benignTest, Malicious, MaliciousTest):
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

    X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train, random_state=21)
    X_test_shuffled, y_test_shuffled = shuffle(X_test, y_test, random_state=21)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train_shuffled)
    X_test_scaled = scaler.transform(X_test_shuffled)

    return X_train_scaled, y_train_shuffled, X_test_scaled, y_test_shuffled

def hyper_para(X_data, Y_data, verb=False):

    param_grid = {
    'C': [0.1, 1, 10, 100, 1000],  
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
    }

    svm_model = svm.SVC(kernel="rbf", verbose=verb)

    # Perform GridSearch with 5-fold cross-validation
    grid_search = GridSearchCV(svm_model, param_grid, refit=True)
    grid_search.fit(X_data, Y_data)

    # Print the best parameters and the best score
    print("Best Parameters: ", grid_search.best_params_)
    print("Best Cross-validation Score: ", grid_search.best_score_)


def create_model(training, labels):
    svm_model = svm.SVC(C=100, kernel="rbf", gamma="scale")

    svm_model.fit(training, labels)

def statistics(svm_model, test_data, test_labels):
    y_prediction = svm_model.predict(test_data)

    confusionMatrix = confusion_matrix(test_labels, y_prediction)
    confusionMatrixDF = pd.DataFrame(confusionMatrix, index=["Actual Benign", "Actual Malicious"], columns=["Predicted Benign", "Predicted Malicious"])
    accuracy = accuracy_score(test_labels, y_prediction)
    recall = recall_score(test_labels, y_prediction)
    precision = precision_score(test_labels, y_prediction)

    print(f"accuracy: {accuracy}")
    print(f"recall: {recall}")
    print(f"precision: {precision}")
    print()
    print("Confusion Matrix" + str(confusionMatrixDF))

def main():
    beignTraining = "l2-benign_training.csv"
    beignTesting = "l2-malicious_training.csv"
    MaliciousTraining = "l2-benign_testing.csv"
    MaliciousTesting = "l2-malicious_testing.csv"

    X_train, y_train, X_test, y_test = format_data(beignTraining, beignTesting, MaliciousTraining, MaliciousTesting)

    #hyper_para(X_train_scaled, y_train_shuffled)

    svm_model = create_model(X_train, y_train)

    statistics(svm_model, X_test, y_test)

if __name__ == '__main__':
    main()