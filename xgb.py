import pandas as pd
import numpy as np
from xgboost import XGBClassifier, train, DMatrix, Booster

from sklearn import preprocessing as prp
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, accuracy_score

from utils import pretty_table, predict_with_threshold, get_xgb_feat_importances

sk_type_classifier = False
preprocess_data = False
preprocess_test = False

train_model = False
model_file = 'models\md100_nr10_lr0.3.model'
threshold = None

header = pd.read_csv('JAT_MCV_VAR_INT_CAL_HIST_VP_LABELS.csv', header=None)
target = 'BMI'

# adhoc for int_cal_hist_vp
feat_names = list(header.iloc[0, range(2, 696)].values)
feat_names[116] = 'rm1'
feat_names[117] = 'rm2'
feat_names[118] = 'rm3'
feat_names[119] = 'rm4'
feat_names[120] = 'rm5'

# adhoc for int_cal_hist_vp
# removes = ['AMORTIZACIONEXIGIBLE', 'AMORTIZACIONNOEXIGIBLE', 'PAGOREALIZADO', 'VOLUNTADPAGO', 'VOLUNTADPAGOPERIODO']
# for rm in removes:
#     features.remove(rm)

strategy = 'mean'
seed = 0

# dataset_folder = 'int_cal_hist'
dataset_folder = 'int_cal_hist_vp'

train_data_file = 'data/{}/train_seed{}_stg-{}.csv'.format(dataset_folder, str(seed), strategy)
val_data_file = 'data/{}/validation_seed{}_stg-{}.csv'.format(dataset_folder, str(seed), strategy)
test_data_file = 'data/{}/test_seed{}_stg-{}.csv'.format(dataset_folder, str(seed), strategy)

if preprocess_data:
    print('------------------ Initialize preprocessing -----------------')
    data = pd.read_csv(
        'JAT_MCV_VAR_INT_CAL_HIST_VP.csv',
        names=['LABEL', 'BMI']+feat_names
    ).drop(
        ['rm1', 'rm2', 'rm3', 'rm4', 'rm5'],
        axis=1
    )
    del feat_names[116], feat_names[116], feat_names[116], feat_names[116], feat_names[116]
    print('------------------ Data Loaded ------------------------------')
    if preprocess_test:
        np.random.seed(seed)
        rand_split = np.random.rand(len(data))
        data = data[rand_split >= 0.99]
    data = data.replace(to_replace='\\N', value=np.nan)

    # select features and target if needed
    data = data.loc[:, [target] + feat_names]
    print('------------------- Impute Data ---------------------')
    # impute nans
    imputer = prp.Imputer(missing_values='NaN', strategy=strategy, axis=0)
    scaler = prp.StandardScaler()
    count = 0
    left_limit = 0
    for right_limit in [200, 400, 689]:
        col_names = feat_names[left_limit:right_limit]
        cols = data.loc[:, col_names]
        imputer = imputer.fit(cols)
        data.loc[:, col_names] = imputer.transform(cols)
        print('Feature name: ', right_limit, count)
        left_limit = right_limit
        count += 1
        # normal distributed

    l_limit = 0
    for r_limit in [200, 400, 689]:
        col_names = feat_names[l_limit:r_limit]
        cols = data.loc[:, col_names]
        data.loc[:, col_names] = scaler.fit_transform(cols)
        print('Feature name: ', r_limit, count)
        l_limit = r_limit
        count += 1

    # split into train, validation and test sets
    np.random.seed(seed)
    rand_split = np.random.rand(len(data))
    train_list = rand_split < 0.6
    val_list = (rand_split >= 0.6) & (rand_split < 0.8)
    test_list = rand_split >= 0.8

    data[train_list].to_csv(train_data_file, index=False)
    data[val_list].to_csv(val_data_file, index=False)
    data[test_list].to_csv(test_data_file, index=False)
    del data
    print('------------------ Finish preprocessing -----------------')

else:
    del feat_names[116], feat_names[116], feat_names[116], feat_names[116], feat_names[116]
print('-------------------- Loading preprocessed data ----------------')
data_train = pd.read_csv(
    'data/train.csv',
    #names=[target]+feat_names
)
data_val = pd.read_csv(
    'data/validation.csv',
    #names=[target]+feat_names
)
data_test = pd.read_csv(
    'data/test.csv',
    #names=[target]+feat_names
)
print('----------------- Finish loading ---------------')

train_y = data_train.loc[:, target].values
train_X = data_train.loc[:, feat_names].values

val_y = data_val.loc[:, target].values
val_X = data_val.loc[:, feat_names].values

test_y = data_test.loc[:, target].values
test_X = data_test.loc[:, feat_names].values

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
    dtrain = DMatrix(train_X, label=train_y, feature_names=feat_names)
    dtest = DMatrix(test_X, label=test_y, feature_names=feat_names)
    dval = DMatrix(val_X, label=val_y, feature_names=feat_names)
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
        feat_imp = get_xgb_feat_importances(bst)
        print(feat_imp)
        print('------------------ Finish trainig -------------------------')
    else:
        bst = Booster({'nthread': 4})
        bst.load_model(model_file)
        feat_imp = get_xgb_feat_importances(bst)
        print(feat_imp)
    # calculate threshold
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