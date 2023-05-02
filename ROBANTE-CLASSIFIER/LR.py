import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Load the heart disease dataset from a CSV file
df = pd.read_csv('heart_disease.csv')

# Replace missing values with NaN
df = df.replace('?', np.nan)

# Drop rows with missing values
df = df.dropna()

# Convert some columns to categorical variables
df['sex'] = df['sex'].astype('category')
df['cp'] = df['cp'].astype('category')
df['fbs'] = df['fbs'].astype('category')
df['restecg'] = df['restecg'].astype('category')
df['exang'] = df['exang'].astype('category')
df['slope'] = df['slope'].astype('category')
df['ca'] = df['ca'].astype('category')
df['thal'] = df['thal'].astype('category')

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

# Split the data into training and testing sets
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a Logistic Regression model and fit it on the training data
model = LogisticRegression(max_iter=10000)

# Perform 5-fold cross-validation and print the accuracy scores
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
cv_results = cross_validate(model, X_train, y_train, cv=5, scoring=scoring)


# Fit the model on the entire training data and predict on the testing data
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Calculate accuracy score on the testing data
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy score
print('Accuracy Score:', accuracy)

# Plot the scatterplot and regression plot of the logistic regression model
sns.regplot(x=model.predict_proba(X_test_scaled)[:,1], y=y_test, logistic=True, scatter_kws={"color": "blue"}, line_kws={"color": "orange"})
plt.title('Logistic Regression Model')
plt.xlabel('Predicted Probability')
plt.ylabel('Actual Outcome')
plt.show()

# Print the classification report and confusion matrix
report = classification_report(y_test, y_pred)
print(report)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)
print("Specificity:", specificity)
print("Sensitivity:", sensitivity)

print("\nCross-validation scores:")
for key in scoring:
    print(key + ":", cv_results["test_" + key])