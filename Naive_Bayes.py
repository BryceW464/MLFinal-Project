import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, f1_score

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

confusionMatrix = confusion_matrix(y_test, y_prediction)
confusionMatrixDF = pd.DataFrame(confusionMatrix, index=["Actual Benign", "Actual Malicious"], columns=["Predicted Benign", "Predicted Malicious"])
accuracy = accuracy_score(y_test, y_prediction)
recall = recall_score(y_test, y_prediction)
precision = precision_score(y_test, y_prediction)
f1Score = f1_score(y_test,y_prediction)

print(f"accuracy: {accuracy}")
print(f"recall: {recall}")
print(f"precision: {precision}")
print("Confusion Matrix" + str(confusionMatrixDF))
print("F1 Score: " + str(f1Score))

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(6, 4))
cax = ax.matshow(confusionMatrix, cmap='Blues')  # Creates the matrix with color
fig.colorbar(cax)  # Adds a color bar to the side
ax.set_xticks(np.arange(2))
ax.set_yticks(np.arange(2))
ax.set_xticklabels(["Predicted Benign", "Predicted Malicious"])
ax.set_yticklabels(["Actual Benign", "Actual Malicious"])

# Rotate the tick labels and set their alignment.
plt.xticks(rotation=45)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Plotting the other metrics (Accuracy, Recall, Precision, F1 Score)
metrics = {'Accuracy': accuracy, 'Recall': recall, 'Precision': precision, 'F1 Score': f1Score}

# Set the Y-axis range between 0.95 and 1.00 for the classification metrics
plt.figure(figsize=(8, 6))
plt.bar(metrics.keys(), metrics.values(), color=['skyblue', 'lightgreen', 'lightcoral', 'lightskyblue'])
plt.title("Classification Metrics")
plt.ylabel("Score")
plt.ylim(0.80, 1.00)

plt.yticks(np.arange(0.80, 1.00, 0.01))

plt.show()
