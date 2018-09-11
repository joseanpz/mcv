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


if preprocess_data:
    print('------------------ Initialize preprocessing -----------------')
    data = pd.read_csv(
        'data/raw/MCV_VAR_1.csv'
    )

    if feature_selection:
        feat_imp_file = 'data/{}/../feature_importances.csv'.format(dataset_folder)
        important_features = pd.read_csv(feat_imp_file)
        feat_names = important_features.loc[:19, 'Feature']
    else:
        feat_names = data.columns[2:517]

    # # revolvente feature
    # revolv_flag_data = data.loc[:, ['REVOLVENTE']]
    # data_revolvente = data.loc[revolv_flag_data['REVOLVENTE'] == 1]
    # data_no_revolvente = data.loc[revolv_flag_data['REVOLVENTE'] == 0]
    # del data
    # data = data_no_revolvente

    print('------------------ Data Loaded ------------------------------')

    if preprocess_test:
        np.random.seed(seed)
        rand_split = np.random.rand(len(data))
        data = data[rand_split >= 0.99]

    # data = data.replace(to_replace='\\N', value=np.nan)

    # select features and target if needed
    # data = data.loc[:, [target] + feat_names]
    print('------------------- Impute Data ---------------------')
    # impute nans
    imputer = prp.Imputer(missing_values='NaN', strategy=strategy, axis=0)
    scaler = prp.RobustScaler()
    count = 0
    left_limit = 0
    for right_limit in [200]:  # [200, 400, 515]:
        col_names = feat_names[left_limit:right_limit]
        cols = data.loc[:, col_names]
        imputer = imputer.fit(cols)
        data.loc[:, col_names] = imputer.transform(cols)
        print('Feature name: ', right_limit, count)
        left_limit = right_limit
        count += 1
        # normal distributed

    l_limit = 0
    for r_limit in [200]:  # [200, 400, 515]:
        col_names = feat_names[l_limit:r_limit]
        cols = data.loc[:, col_names]
        data.loc[:, col_names] = scaler.fit_transform(cols)
        print('Feature name: ', r_limit, count)
        l_limit = r_limit
        count += 1

    # split into train, validation and test sets
    np.random.seed(seed)
    rand_split = np.random.rand(len(data))
    train_list = rand_split < 0.8
    val_list = (rand_split >= 0.6) & (rand_split < 0.8)
    test_list = rand_split >= 0.8

    data[train_list].to_csv(train_data_file, index=False)
    data[val_list].to_csv(val_data_file, index=False)
    data[test_list].to_csv(test_data_file, index=False)
    del data
    print('------------------ Finish preprocessing -----------------')

# else:
#     header = pd.read_csv('data/raw/JAT_MCV_VAR_INT_CAL_HIST_VP_LABELS.csv', header=None)
#
#     feat_names = list(header.iloc[0, range(2, 522)].values)
#
#     del feat_names[116], feat_names[116], feat_names[116], feat_names[116], feat_names[116]


print('-------------------- Loading preprocessed data ----------------')
data_train = pd.read_csv(
    train_data_file
)
data_val = pd.read_csv(
    val_data_file
)
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
    evals = [(dtest, 'eval')]
    num_round = 50
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
        bst = train(params, dtrain, num_round, evals, feval=val_func)

        bst.dump_model(
            'models/{}/dumps/md{}_nr{}_lr{}.model.dump.raw.txt'.format(
                dataset_folder,
                str(max_depth),
                str(num_round),
                str(learning_rate)
            ),
            with_stats=False
        )
        # booster pickle dump
        pickle.dump(
            bst,
            open(
                'models/{}/pickles/md{}_nr{}_lr{}.model'.format(
                    dataset_folder,
                    str(max_depth),
                    str(num_round),
                    str(learning_rate)
                ),
                "wb"
            )
        )

        feat_imp = get_xgb_feat_importances(bst)
        feat_imp.to_csv('data/{}/feature_importances.csv'.format(dataset_folder), index=False)
        print(feat_imp)
        print('------------------ Finish trainig -------------------------')
    else:
        old_bst = Booster({'nthread': 4})
        old_bst.load_model(
            'models/{}/md{}_nr{}_lr{}.model'.format(
                dataset_folder,
                str(max_depth),
                str(num_round),
                str(learning_rate)
            )
        )

        # booster pickle load
        bst = pickle.load(
            open(
                'models/{}/pickles/md{}_nr{}_lr{}.model'.format(
                    dataset_folder,
                    str(max_depth),
                    str(num_round),
                    str(learning_rate)
                ),
                "rb"
            )
        )

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