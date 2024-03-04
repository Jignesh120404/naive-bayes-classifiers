import pandas as pd
import numpy as np
import csv
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

### Loading dataset ###
df = pd.read_csv('Iris.csv')
df.drop('Id', axis=1, inplace=True)

columns = df.columns.tolist()
Classes_ = df['Species'].unique().tolist()

# Calculate prior probability for each class
prior_probs = {}
total_instances = len(df)
for class_ in Classes_:
    class_instances = len(df[df['Species'] == class_])
    prior_probs[class_] = class_instances / total_instances

def get_pdf_par(attribute, class_):
    result = df.loc[df['Species'] == class_, attribute]
    mean_ = np.mean(result)
    var_ = np.std(result) ** 2
    return mean_, var_

def get_likelihood(value_, mean_, var_):
    r_ = (1 / np.sqrt(2 * np.pi * var_)) * (np.exp((-(value_ - mean_) ** 2) / (2 * var_)))
    return r_

# Predict classes for each row
S_ = []
predicted_classes = []
for _, row in df.iterrows():
    L = []
    for j in Classes_:
        prior_prob = prior_probs[j]
        L_ = np.log(prior_prob)  # Initialize with the log of prior probability
        for i in columns:
            if i != 'Species':  # Exclude species column from likelihood calculation
                m_, v_ = get_pdf_par(i, j)
                likelihood_ = np.log(get_likelihood(row[i], m_, v_))
                L_ += likelihood_
        L.append(L_)
    predicted_class = Classes_[np.argmax(L)]
    predicted_classes.append(predicted_class)

# Get actual species labels from the dataset
actual_classes = df['Species'].tolist()

# Create confusion matrix
conf_matrix = confusion_matrix(actual_classes, predicted_classes, labels=Classes_)
print(conf_matrix)
# Plot confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=Classes_, yticklabels=Classes_)
# plt.xlabel('Predicted Class')
# plt.ylabel('Actual Class')
# plt.title('Confusion Matrix')
# plt.show()
