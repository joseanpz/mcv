import pandas as pd
import numpy as np
from xgboost import XGBClassifier, train, DMatrix, Booster
import pickle

from bayes_opt import BayesianOptimization

from sklearn import preprocessing as prp
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, accuracy_score

from utils import pretty_table, predict_with_threshold, get_xgb_feat_importances, val_func

from bayes_opt import BayesianOptimization

pd.options.display.max_rows = 4000

sk_type_classifier = False
preprocess_data = False
preprocess_test = False

feature_selection = True

train_model = True
# model_file = 'models\md100_nr10_lr0.3.model'
threshold = None
target = 'BMI'

strategy = 'median'
seed = 0

# dataset_folder = 'int_cal_hist'
dataset_folder = 'mcv_var_1/robust_scaler/feature_selection'

train_data_file = 'data/{}/../train_seed{}_stg-{}.csv'.format(dataset_folder, str(seed), strategy)
val_data_file = 'data/{}/../validation_seed{}_stg-{}.csv'.format(dataset_folder, str(seed), strategy)
test_data_file = 'data/{}/../test_seed{}_stg-{}.csv'.format(dataset_folder, str(seed), strategy)

print('----------------- Loading data ---------------')
data_train = pd.read_csv(
    train_data_file
)

np.random.seed(seed)
rand_split = np.random.rand(len(data_train))
val_list = rand_split >= 0.75
# train_list = rand_split < 0.75

data_val = data_train[val_list]
# data_train = data_train[train_list]
data_test = pd.read_csv(
    test_data_file
)
if feature_selection:
    feat_imp_file = 'data/{}/../feature_importances.csv'.format(dataset_folder)
    important_features = pd.read_csv(feat_imp_file)
    feat_names = important_features.loc[:49, 'Feature']
else:
    feat_names = data_train.columns[2:517]

print('----------------- Finish loading ---------------')

train_y = data_train.loc[:, target].values
train_X = data_train.loc[:, feat_names].values

val_y = data_val.loc[:, target].values
val_X = data_val.loc[:, feat_names].values

test_y = data_test.loc[:, target].values
test_X = data_test.loc[:, feat_names].values

print('----------------  Initialize DMatrix -------------------')
dtrain = DMatrix(train_X, label=train_y, feature_names=feat_names)
dtest = DMatrix(test_X, label=test_y, feature_names=feat_names)
dval = DMatrix(val_X, label=val_y, feature_names=feat_names)
print('----------------  Finish Init Dmatrix -------------------')

# specify parameters via map
evals = [(dtest, 'eval')]
num_round = 50
max_depth = 10
learning_rate = 0.3
params_tmp = {
    'max_depth': max_depth,
    'eta': learning_rate,
    'silent': 1,
    'objective': 'binary:logistic',
    "eval_metric": "auc",
    "seed": "1",
    "scale_pos_weight": 24
}
params = {
    'max_depth': 12,
    'eta': 0.1,
    'silent': 1,
    'objective': 'binary:logistic',
    "eval_metric": "auc",
    "seed": "1",
    "scale_pos_weight": 24,
    "max_delta_step": 2.4592238788322622,
    "min_child_weight": 10.108909877670685,
    "gamma": 0.001,
    "colsample_bytree": 0.4
}
# learning_rate = 0.1,
# n_estimators = 200,
# max_depth = 12,
# min_child_weight = 10.108909877670685,
# gamma = 0.001,
# subsample = 1,
# colsample_bytree = 0.4,
# objective = 'binary:logistic',
# # nthread=4,
# max_delta_step = 2.4592238788322622,
# scale_pos_weight = 1,
# seed = 1)

bst = pickle.load(
    open('models/{}/pickles/md10_nr150_lr0.3_aux.model'.format(dataset_folder), "rb")
)

print('------------------ Initialize trainig ---------------------')
bst = train(params, dtrain, num_round, evals, feval=val_func, xgb_model=bst)

bst.dump_model(
    'models/{}/dumps/md{}_nr{}_lr{}.model_aux.dump.raw.txt'.format(
        dataset_folder, str(max_depth), str(bst.best_ntree_limit), str(learning_rate)
    ),
    with_stats=False
)
# booster pickle dump
pickle.dump(
    bst,
    open(
        'models/{}/pickles/md{}_nr{}_lr{}_aux.model'.format(
            dataset_folder,
            str(max_depth),
            str(bst.best_ntree_limit),
            str(learning_rate)
        ),
        "wb"
    )
)

feat_imp = get_xgb_feat_importances(bst)
feat_imp.to_csv('data/{}/feature_importances.csv'.format(dataset_folder), index=False)
print(feat_imp)
print('------------------ Finish trainig -------------------------')

if not threshold:
    print('------------------ Intitialize threshold calculation -----------------')
    f1_sc = 0
    max_step = 0
    thr_sample = (dtest, test_y)
    _score_preds = bst.predict(thr_sample[0])
    for thr_step in np.linspace(0, 1, 101):
        _preds = predict_with_threshold(_score_preds, thr_step)
        f1_sc_step = f1_score(thr_sample[1], _preds)
        print(thr_step, f1_sc_step)
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
    ['accuracy score', accuracy_score(train_y, train_pred), accuracy_score(val_y, val_pred),
     accuracy_score(test_y, test_pred)],
    ['recall score', recall_score(train_y, train_pred), recall_score(val_y, val_pred), recall_score(test_y, test_pred)],
    ['precision_score', precision_score(train_y, train_pred), precision_score(val_y, val_pred),
     precision_score(test_y, test_pred)]
]

# print pretty table
pretty_table(summary)
print(params)
print('num_round:', bst.best_ntree_limit)