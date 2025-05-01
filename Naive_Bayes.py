import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Loading data
#column 35 for label
benignTrainingData = pd.read_csv("l2-benign_training.csv")
maliciousTrainingData = pd.read_csv("l2-malicious_training.csv")
benignTestingData = pd.read_csv("l2-benign_testing.csv")
maliciousTestingData = pd.read_csv("l2-malicious_testing.csv")

benignTrainingData['label'] = 0
maliciousTrainingData['label'] = 1
benignTestingData['label'] = 0
maliciousTestingData['label'] = 1

combinedTrainingData = pd.concat([benignTrainingData, maliciousTrainingData], ignore_index=True)
combinedTestingData = pd.concat([benignTestingData, maliciousTestingData], ignore_index=True)

X_train = combinedTrainingData.drop(columns=[combinedTrainingData.columns[-1]])
y_train = combinedTrainingData[combinedTrainingData[combinedTrainingData].columns[-1]]
X_test = combinedTestingData.drop(columns=[combinedTestingData.columns[-1]])
y_test = combinedTestingData[combinedTestingData.columns[-1]]

gnb_model = GaussianNB()
gnb_model.fit(X_train, y_train)

y_prediction = gnb_model.predict(X_test)

accuracy = accuracy_score(y_test, y_prediction)

print(f"accuracy: {accuracy}")
