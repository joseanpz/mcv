import pandas as pd
import numpy as np
from xgboost import XGBClassifier, train, DMatrix

from sklearn import preprocessing as prp
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, accuracy_score

from utils import pretty_table, predict_with_threshold

data = pd.read_csv('titanic3.csv')

target = 'survived'
features = ['pclass', 'age', 'sibsp', 'parch', 'fare']
strategy = 'mean'

# select features and target if needed
data = data.loc[:, [target] + features]

# impute nans
imputer = prp.Imputer(missing_values='NaN', strategy=strategy, axis=0)
imputer = imputer.fit(data.loc[:, features])
data.loc[:, features] = imputer.transform(data.loc[:, features])

# normal distributed
data.loc[:, features] = prp.StandardScaler().fit_transform(data.drop(target, axis=1))

# split into train, validation and test sets
np.random.seed(0)
rand_split = np.random.rand(len(data))
train_list = rand_split < 0.8
val_list = (rand_split >= 0.8) & (rand_split < 0.9)
test_list = rand_split >= 0.9

data_train = data[train_list]
data_val = data[val_list]
data_test = data[test_list]

train_y = data_train.loc[:, target].values
train_X = data_train.loc[:, features].values

val_y = data_val.loc[:, target].values
val_X = data_val.loc[:, features].values

test_y = data_test.loc[:, target].values
test_X = data_test.loc[:, features].values

# describe training data
print(data_train.describe())
print(data_train.head(), '\n', '-----------------')


# describe validation data
print(data_val.describe())
print(data_val.head(), '\n', '-----------------')

# describe test data
print(data_test.describe())
print(data_test.head(), '\n', '-----------------')

# fit xgboost
classifier = XGBClassifier(max_depth=5, learning_rate=0.2, nthread=4)
classifier.fit(train_X, train_y)

train_pred = classifier.predict(train_X)
val_pred = classifier.predict(val_X)
test_pred = classifier.predict(test_X)

cm_train = confusion_matrix(train_y, train_pred)
cm_val = confusion_matrix(val_y, val_pred)
cm_test = confusion_matrix(test_y, test_pred)

cm_train_pct = cm_train / cm_train.astype(np.float).sum()
cm_val_pct = cm_val / cm_val.astype(np.float).sum()
cm_test_pct = cm_test / cm_test.astype(np.float).sum()

summary = [
    ['------', 'Train', 'Validation', 'Test'],
    ['confusion matrix', cm_train, cm_val, cm_test],
    ['confusion matrix pct', cm_train_pct, cm_val_pct, cm_test_pct],
    ['f1_score', f1_score(train_y, train_pred), f1_score(val_y, val_pred), f1_score(test_y, test_pred)],
    ['accuracy score', accuracy_score(train_y, train_pred), accuracy_score(val_y, val_pred), accuracy_score(test_y, test_pred)],
    ['recall score', recall_score(train_y, train_pred), recall_score(val_y, val_pred), recall_score(test_y, test_pred)],
    ['precision_score', precision_score(train_y, train_pred), precision_score(val_y, val_pred), precision_score(test_y, test_pred)]
]

# print pretty table
pretty_table(summary)



# read in data
dtrain = DMatrix(train_X, label=train_y)
dtest = DMatrix(test_X, label=test_y)
dval = DMatrix(val_X, label=val_y)

# specify parameters via map
params = {'max_depth': 3, 'eta': 0.21, 'silent': 1, 'objective': 'binary:logistic' }
num_round = 100
bst = train(params, dtrain, num_round)

# calculate threshold
f1_sc = 0
max_step = 0
thr_sample = (dval, val_y)
_score_preds = bst.predict(thr_sample[0])
for thr_step in np.linspace(0, 1, 101):
    _preds = predict_with_threshold(_score_preds, thr_step)
    f1_sc_step = f1_score(thr_sample[1], _preds)
    # print(thr_step, f1_sc_step)
    if f1_sc_step >= f1_sc:
        f1_sc = f1_sc_step
        max_step = thr_step
threshold = max_step
print('threshold', threshold)

# make predictions on datasets
train_pred = predict_with_threshold(bst.predict(dtrain), threshold)
val_pred = predict_with_threshold(bst.predict(dval), threshold)
test_pred = predict_with_threshold(bst.predict(dtest), threshold)

cm_train = confusion_matrix(train_y, train_pred)
cm_val = confusion_matrix(val_y, val_pred)
cm_test = confusion_matrix(test_y, test_pred)

cm_train_pct = cm_train / cm_train.astype(np.float).sum()
cm_val_pct = cm_val / cm_val.astype(np.float).sum()
cm_test_pct = cm_test / cm_test.astype(np.float).sum()

summary = [
    ['------', 'Train', 'Validation', 'Test'],
    ['confusion matrix', cm_train, cm_val, cm_test],
    ['confusion matrix pct', cm_train_pct, cm_val_pct, cm_test_pct],
    ['f1_score', f1_score(train_y, train_pred), f1_score(val_y, val_pred), f1_score(test_y, test_pred)],
    ['accuracy score', accuracy_score(train_y, train_pred), accuracy_score(val_y, val_pred), accuracy_score(test_y, test_pred)],
    ['recall score', recall_score(train_y, train_pred), recall_score(val_y, val_pred), recall_score(test_y, test_pred)],
    ['precision_score', precision_score(train_y, train_pred), precision_score(val_y, val_pred), precision_score(test_y, test_pred)]
]

# print pretty table
pretty_table(summary)


print('finish!')