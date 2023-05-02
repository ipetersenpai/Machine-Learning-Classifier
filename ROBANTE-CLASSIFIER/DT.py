import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the heart disease dataset
heart_df = pd.read_csv("heart_disease.csv")

# Split the dataset into training and testing sets
X = heart_df.drop('target', axis=1)
y = heart_df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree classifier
dtc = DecisionTreeClassifier()

# Calculate accuracy using cross-validation with 5 folds
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
cv_results = cross_validate(dtc, X_train, y_train, cv=5, scoring=scoring)

# Train the classifier on the training set
dtc.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = dtc.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Make predictions on a new test set
new_test_df = pd.read_csv("newset.csv")
new_X_test = new_test_df.drop('target', axis=1)
new_y_pred = dtc.predict(new_X_test)
print("Target Values: ", new_y_pred[0])

# Print classification report and confusion matrix
report = classification_report(y_test, y_pred)
print(report)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)
print("Specificity:", specificity)
print("Sensitivity:", sensitivity)

# Print cross-validation results
print("\nCross-validation scores:")
for key in scoring:
    print(key + ":", cv_results["test_" + key])