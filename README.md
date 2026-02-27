# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Dataset Loading & Preparation Load the Iris dataset using load_iris(). Convert it into a Pandas DataFrame. Separate features (X) and target labels (y).
2.Split the dataset into training and testing sets using train_test_split(). Use 80% data for training and 20% for testing. 3.Create an SGDClassifier with specified parameters. Train the classifier using the 3.training data (fit() method).
4.Predict class labels for test data. Calculate accuracy using accuracy_score(). Generate the confusion matrix to evaluate classification performance. 


## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: S.Risitha
RegisterNumber:25018977  
*/ /*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: S.Risitha
RegisterNumber:25018977
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# Load the Iris dataset
iris = load_iris()
# Create a Pandas DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
# Display the first few rows of the dataset
print(df.head())
# Split the data into features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
random_state=42)
# Create an SGD classifier with default parameters
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)
# Train the classifier on the training data
sgd_clf.fit(X_train, y_train)
# Make predictions on the testing data
y_pred = sgd_clf.predict(X_test)
# Evaluate the classifier's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)  

*/
```

## Output:
![WhatsApp Image 2026-02-27 at 14 00 55](https://github.com/user-attachments/assets/1c59d16c-b902-457c-9f6b-41743fb7330f)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
