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
model = RandomForestClassifier(n_estimators=100,
                               random_state=RSEED,
                               max_features='sqrt',
                               n_jobs=-1, verbose=1)

# Fit on training data
model.fit(train, train_labels)

n_nodes = []
max_depths = []

# Stats about the trees in random forest
for ind_tree in model.estimators_:
    n_nodes.append(ind_tree.tree_.node_count)
    max_depths.append(ind_tree.tree_.max_depth)

print(f'Average number of nodes {int(np.mean(n_nodes))}')
print(f'Average maximum depth {int(np.mean(max_depths))}')

# Training predictions (to demonstrate overfitting) - from tutorial
train_rf_predictions = model.predict(train)
train_rf_probs = model.predict_proba(train)[:, 1]

# Testing predictions (to determine performance) - from tutorial
rf_predictions = model.predict(test)
rf_probs = model.predict_proba(test)[:, 1]

# Plot formatting
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 18


# method to evaluate the model

def evaluate_model(predictions, probs, train_predictions, train_probs):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""

    baseline = {}

    baseline['recall'] = recall_score(test_labels,
                                      [1 for _ in range(len(test_labels))])
    baseline['precision'] = precision_score(test_labels,
                                            [1 for _ in range(len(test_labels))])
    baseline['roc'] = 0.5

    results = {}

    results['recall'] = recall_score(test_labels, predictions)
    results['precision'] = precision_score(test_labels, predictions)
    results['roc'] = roc_auc_score(test_labels, probs)

    train_results = {}
    train_results['recall'] = recall_score(train_labels, train_predictions)
    train_results['precision'] = precision_score(train_labels, train_predictions)
    train_results['roc'] = roc_auc_score(train_labels, train_probs)

    for metric in ['recall', 'precision', 'roc']:
        print(
            f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')

    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(test_labels, [1 for _ in range(len(test_labels))])
    model_fpr, model_tpr, _ = roc_curve(test_labels, probs)

    plt.figure(figsize=(8, 6))
    plt.rcParams['font.size'] = 16

    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label='baseline')
    plt.plot(model_fpr, model_tpr, 'r', label='model')
    plt.legend();
    plt.xlabel('False Positive Rate');
    plt.ylabel('True Positive Rate');
    plt.title('ROC Curves');
    plt.show();

evaluate_model(rf_predictions, rf_probs, train_rf_predictions, train_rf_probs)
plt.savefig('roc_auc_curve.png')
