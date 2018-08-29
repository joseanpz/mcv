import pandas as pd
import numpy as np
from xgboost import XGBClassifier, train, DMatrix, Booster

from sklearn import preprocessing as prp
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, accuracy_score

from utils import pretty_table, predict_with_threshold

sk_type_classifier = False
preprocess_data = False
preprocess_test = False

train_model = False
model_file = 'models\md100_nr10_lr0.3.model'
threshold = 0.10

header = pd.read_csv('JAT_MCV_UNIVERSO_MODELADO_PREVIO_labels.csv', header=None)
target = 'BMI'
features = list(header.iloc[0, range(2, 522)].values)
removes = ['AMORTIZACIONEXIGIBLE', 'AMORTIZACIONNOEXIGIBLE', 'PAGOREALIZADO', 'VOLUNTADPAGO', 'VOLUNTADPAGOPERIODO']
for rm in removes:
    features.remove(rm)
strategy = 'mean'

if preprocess_data:
    print('------------------ Initialize preprocessing -----------------')
    data = pd.read_csv('JAT_MCV_UNIVERSO_MODELOADO_FILTROS.csv', header=None, names=header.loc[0, :].values)
    print('------------------ Data Loaded ------------------------------')
    if preprocess_test:
        np.random.seed(0)
        rand_split = np.random.rand(len(data))
        data = data[rand_split >= 0.99]
    data = data.replace(to_replace='\\N', value=np.nan)

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
    train_list = rand_split < 0.6
    val_list = (rand_split >= 0.6) & (rand_split < 0.8)
    test_list = rand_split >= 0.8

    data[train_list].to_csv('train.csv', index=False)
    data[val_list].to_csv('validation.csv', index=False)
    data[test_list].to_csv('test.csv', index=False)
    del data
    print('------------------ Finish preprocessing -----------------')


print('-------------------- Loading preprocessed data ----------------')
data_train = pd.read_csv('train.csv')
data_val = pd.read_csv('validation.csv')
data_test = pd.read_csv('test.csv')
print('----------------- Finish loading ---------------')

train_y = data_train.loc[:, target].values
train_X = data_train.loc[:, features].values

val_y = data_val.loc[:, target].values
val_X = data_val.loc[:, features].values

test_y = data_test.loc[:, target].values
test_X = data_test.loc[:, features].values

# describe training data
# print(data_train.describe())
# print(data_train.head(), '\n', '-----------------')


# describe validation data
# print(data_val.describe())
# print(data_val.head(), '\n', '-----------------')

# describe test data
# print(data_test.describe())
# print(data_test.head(), '\n', '-----------------')


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
    print('----------------  Finish Init Dmatrix -------------------')

    # specify parameters via map
    num_round = 100
    max_depth = 10
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
        bst.save_model('models/md{}_nr{}_lr{}.model'.format(str(num_round), str(max_depth), str(learning_rate)))
        print('------------------ Finish trainig -------------------------')
    else:
        bst = Booster({'nthread': 4})
        bst.load_model(model_file)
    # calculate threshold
    if not threshold:
        print('------------------ Intitialize threshold calcul -----------------')
        f1_sc = 0
        max_step = 0
        thr_sample = (dtest, test_y)
        _score_preds = bst.predict(thr_sample[0])
        for thr_step in np.linspace(0, 1, 101):
            _preds = predict_with_threshold(_score_preds, thr_step)
            f1_sc_step = f1_score(thr_sample[1], _preds)
            # print(thr_step, f1_sc_step)
            if f1_sc_step >= f1_sc:
                f1_sc = f1_sc_step
                max_step = thr_step
        threshold = max_step
        print('----------- threshold----------- \n', threshold)

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