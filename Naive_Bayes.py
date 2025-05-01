import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def load_dataset(filename, label):
    dataframe = pd.read_csv(filename)
    dataframe.drop(dataframe.columns[[0, 1, 4, 34]], axis=1, inplace=True)

    dataframe['label'] = label

    dataframe.dropna(inplace=True)

    return dataframe

benignTrainingData = load_dataset("l2-benign_training.csv", 0)
maliciousTrainingData = load_dataset("l2-malicious_training.csv", 1)
benignTestingData = load_dataset("l2-benign_testing.csv", 0)
maliciousTestingData = load_dataset("l2-malicious_testing.csv", 1)

combinedTrainingData = pd.concat([benignTrainingData, maliciousTrainingData])
combinedTestingData = pd.concat([benignTestingData, maliciousTestingData])

X_train = combinedTrainingData.drop(columns=['label'])
y_train = combinedTrainingData['label']
X_test = combinedTestingData.drop(columns=['label'])
y_test = combinedTestingData['label']

gnb_model = GaussianNB()
gnb_model.fit(X_train, y_train)

y_prediction = gnb_model.predict(X_test)

confusionMatrix = (y_test, y_prediction)
accuracy = accuracy_score(y_test, y_prediction)

print(f"accuracy: {accuracy}")
