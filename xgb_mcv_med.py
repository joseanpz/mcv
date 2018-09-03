import pandas as pd
import numpy as np
from xgboost import XGBClassifier, train, DMatrix, Booster

from sklearn import preprocessing as prp
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, accuracy_score

from utils import pretty_table, predict_with_threshold

sk_type_classifier = False
preprocess_data = True
preprocess_test = False

train_model = True
model_file = 'models\md100_nr10_lr0.3.model'
model_file2 = None
threshold = None
threshold2 = None

strategy = 'median'

target1 = 'CV_6M'
target2 = '30+_6M'

if preprocess_data:
    print('------------------ Initialize preprocessing -----------------')
    df1 = pd.read_csv('data/med/CV_6M.csv')
    df2 = pd.read_csv('data/med/30+_6M.csv')
    print('------------------ Data Loaded ------------------------------')
    if preprocess_test:
        np.random.seed(0)
        rand_split = np.random.rand(len(df1))
        df1 = df1[rand_split >= 0.99]

    # impute nans
    imputer1 = prp.Imputer(missing_values='NaN', strategy=strategy, axis=0)
    imputer1 = imputer1.fit(df1.iloc[:, 1:])
    df1.iloc[:, 1:] = imputer1.transform(df1.iloc[:, 1:])

    imputer2 = prp.Imputer(missing_values='NaN', strategy=strategy, axis=0)
    imputer2 = imputer2.fit(df2.iloc[:, 1:])
    df2.iloc[:, 1:] = imputer2.transform(df2.iloc[:, 1:])

    # normal distributed
    df1.iloc[:, 1:] = prp.StandardScaler().fit_transform(df1.iloc[:, 1:])
    df2.iloc[:, 1:] = prp.StandardScaler().fit_transform(df2.iloc[:, 1:])


    # split into train, validation and test sets
    np.random.seed(0)
    rand_split = np.random.rand(len(df1))
    train_list = rand_split < 0.6
    val_list = (rand_split >= 0.6) & (rand_split < 0.8)
    test_list = rand_split >= 0.8

    df1[train_list].to_csv('data/med/train.csv', index=False)
    df1[val_list].to_csv('data/med/validation.csv', index=False)
    df1[test_list].to_csv('data/med/test.csv', index=False)

    df2[train_list].to_csv('data/med/train2.csv', index=False)
    df2[val_list].to_csv('data/med/validation2.csv', index=False)
    df2[test_list].to_csv('data/med/test2.csv', index=False)
    # del data
    print('------------------ Finish preprocessing -----------------')


print('-------------------- Loading preprocessed data ----------------')
data_train = pd.read_csv('data/med/train.csv')
data_val = pd.read_csv('data/med/validation.csv')
data_test = pd.read_csv('data/med/test.csv')

data_train2 = pd.read_csv('data/med/train2.csv')
data_val2 = pd.read_csv('data/med/validation2.csv')
data_test2 = pd.read_csv('data/med/test2.csv')
print('----------------- Finish loading ---------------')

train_y = data_train.loc[:, target1].values
train_X = data_train.iloc[:, 1:].values

val_y = data_val.loc[:, target1].values
val_X = data_val.iloc[:, 1:].values

test_y = data_test.loc[:, target1].values
test_X = data_test.iloc[:, 1:].values

train_y2 = data_train2.loc[:, target2].values
train_X2 = data_train2.iloc[:, 1:].values

val_y2 = data_val2.loc[:, target2].values
val_X2 = data_val2.iloc[:, 1:].values

test_y2 = data_test2.loc[:, target2].values
test_X2 = data_test2.iloc[:, 1:].values


# fit xgboost
if sk_type_classifier:
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

else:
    # read in data

    print('----------------  Initialize DMatrix -------------------')
    dtrain = DMatrix(train_X, label=train_y)
    dtest = DMatrix(test_X, label=test_y)
    dval = DMatrix(val_X, label=val_y)

    dtrain2 = DMatrix(train_X2, label=train_y2)
    dtest2 = DMatrix(test_X2, label=test_y2)
    dval2 = DMatrix(val_X2, label=val_y2)
    print('----------------  Finish Init Dmatrix -------------------')

    # specify parameters via map
    num_round = 100
    max_depth = 15
    learning_rate = 0.3
    params = {
        'max_depth': max_depth,
        'eta': learning_rate,
        'silent': 1,
        'objective': 'binary:logistic',
        "eval_metric": "auc",
        "seed": "1"
    }

    if train_model:
        print('------------------ Initialize trainig ---------------------')
        bst = train(params, dtrain, num_round)
        bst.save_model('models/med/md{}_nr{}_lr{}.model'.format(str(num_round), str(max_depth), str(learning_rate)))
        bst2 = train(params, dtrain2, num_round)
        bst2.save_model('models/med/2md{}_nr{}_lr{}.model'.format(str(num_round), str(max_depth), str(learning_rate)))
        print('------------------ Finish trainig -------------------------')
    else:
        bst = Booster({'nthread': 4})
        bst.load_model(model_file)
        bst2 = Booster({'nthread': 4})
        bst2.load_model(model_file2)
    # calculate threshold
    if not threshold:
        print('------------------ Intitialize threshold calculation -----------------')
        f1_sc = 0
        max_step = 0
        thr_sample = (dtest, test_y)
        val_func = f1_score
        _score_preds = bst.predict(thr_sample[0])
        for thr_step in np.linspace(0, 1, 101):
            _preds = predict_with_threshold(_score_preds, thr_step)
            f1_sc_step = val_func(thr_sample[1], _preds)
            # print(thr_step, f1_sc_step)
            if f1_sc_step >= f1_sc:
                f1_sc = f1_sc_step
                max_step = thr_step
        threshold = max_step
        print('----------- threshold----------- \n', threshold)
    if not threshold2:
        print('------------------ Intitialize threshold calculation -----------------')
        f1_sc = 0
        max_step = 0
        thr_sample = (dtest2, test_y2)
        val_func = f1_score
        _score_preds = bst2.predict(thr_sample[0])
        for thr_step in np.linspace(0, 1, 101):
            _preds = predict_with_threshold(_score_preds, thr_step)
            f1_sc_step = val_func(thr_sample[1], _preds)
            # print(thr_step, f1_sc_step)
            if f1_sc_step >= f1_sc:
                f1_sc = f1_sc_step
                max_step = thr_step
        threshold2 = max_step
        print('----------- threshold2----------- \n', threshold2)

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

    # make predictions on datasets2
    train_pred2 = predict_with_threshold(bst2.predict(dtrain2), threshold2)
    val_pred2 = predict_with_threshold(bst2.predict(dval2), threshold2)
    test_pred2 = predict_with_threshold(bst2.predict(dtest2), threshold2)

    cm_train2 = confusion_matrix(train_y2, train_pred2)
    cm_val2 = confusion_matrix(val_y2, val_pred2)
    cm_test2 = confusion_matrix(test_y2, test_pred2)

    cm_train_pct2 = cm_train2 / cm_train2.astype(np.float).sum()
    cm_val_pct2 = cm_val2 / cm_val2.astype(np.float).sum()
    cm_test_pct2 = cm_test2 / cm_test2.astype(np.float).sum()

    summary2 = [
        ['------', 'Train', 'Validation', 'Test'],
        ['confusion matrix', cm_train2, cm_val2, cm_test2],
        ['confusion matrix pct', cm_train_pct2, cm_val_pct2, cm_test_pct2],
        ['f1_score', f1_score(train_y2, train_pred2), f1_score(val_y2, val_pred2), f1_score(test_y2, test_pred2)],
        ['accuracy score', accuracy_score(train_y2, train_pred2), accuracy_score(val_y2, val_pred2),
         accuracy_score(test_y2, test_pred2)],
        ['recall score', recall_score(train_y2, train_pred2), recall_score(val_y2, val_pred2),
         recall_score(test_y2, test_pred2)],
        ['precision_score', precision_score(train_y2, train_pred2), precision_score(val_y2, val_pred2),
         precision_score(test_y2, test_pred2)]
    ]

    # print pretty table
    pretty_table(summary)
    print('-------------------------------------------------------------------------')
    # print pretty table
    pretty_table(summary2)
    print(params)
    print('num_round:', num_round)

# confusion matrix:
# tn  fp
# fn  tp
# -------
# f1 score: tp / (tp + (fp + fn)/2)
# -------
# acuracy score: (tp + tn) / (tp + fp + tn + fn)
# -------
# recall score: tp / (tp + fn)
# -------
# precision score: tp / (tp + fp)


print('finish!')