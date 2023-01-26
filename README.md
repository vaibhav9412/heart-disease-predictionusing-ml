# heart-disease-predictionusing-ml
This code uses the Random Forest algorithm to predict heart disease. The dataset is loaded, split into features and target, and then split into training and test sets. The classifier is trained on the training data and predictions are made on the test data. Finally, the accuracy of the model is calculated.


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv("heart_disease_data.csv")

# Split the data into features and target
X = data.drop("target", axis=1)
y = data["target"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a random forest classifier
clf = RandomForestClassifier()

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Calculate the accuracy of the model
acc = accuracy_score(y_test, y_pred)

print("Accuracy: ", acc)
