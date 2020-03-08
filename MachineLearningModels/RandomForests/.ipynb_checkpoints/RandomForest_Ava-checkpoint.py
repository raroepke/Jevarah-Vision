import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd

dataset = pd.read_csv('C:/Users/ava/Desktop/KSU/Year 4/Spring 2020/Senior Project/normalized_data2.csv')
print(dataset.shape)
dataset = dataset.dropna()
print(dataset.shape)
# number of random to choose from
RSEED = 50

model = RandomForestClassifier(n_estimators=100,
                               bootstrap=True,
                               max_features='sqrt')

labels = np.array(dataset.pop('diagnosis'))

imageIds = np.array(dataset.pop('image_id'))

train, test, train_labels, test_labels = train_test_split(dataset,
                                                          labels,
                                                          stratify=labels,
                                                          test_size=0.3,
                                                          random_state=RSEED)
# commented out bc not needed....
# Imputation of missing values
# train = train.fillna(train.mean())
# test = test.fillna(test.mean())

# Create the model with 100 trees
tree = RandomForestClassifier(n_estimators=100,
                               random_state=RSEED,
                               max_features='sqrt',
                               n_jobs=-1, verbose=1)


tree.fit(train, train_labels)

# Make probability predictions
train_probs = tree.predict_proba(train)[:, 1]
probs = tree.predict_proba(test)[:, 1]

train_predictions = tree.predict(train)
predictions = tree.predict(test)

# Calculate the absolute errors
# positive = not nevus
# negative = nevus

# fp: guessed not nevus, but is nevus
# fn: guessed nevus, but is not nevus
# tp: is not nevus
# tn: nevus

errors = 0
fp = 0 
fn = 0
tp = 0
tn = 0
for i in range(len(predictions)):
    # Error
    if predictions[i] != test_labels[i]:
        errors = errors + 1
        if predictions[i] == 'nevus':
            fn = fn + 1
        else:
            fp = fp + 1
    # All
    if test_labels[i] == 'nevus':
        tn = tn + 1
    else:
        tp = tp + 1


# errors = abs(predictions - test_labels)
# # Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# # Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / len(test_labels))
# # Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

print('False Negative Rate:', round(fn / (fn + tp), 2), '%.')