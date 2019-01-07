import pandas as pd
import numpy as np
import pickle

from bayes_opt import BayesianOptimization

from sklearn.svm import SVC
from sklearn import preprocessing as prp
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, accuracy_score, classification_report

from utils import pretty_table, predict_with_threshold, get_xgb_feat_importances, val_func, sampler

from bayes_opt import BayesianOptimization
pd.options.display.max_rows = 4000

data_preprocessed_file = 'data/mcv_var_1/preprocessed_data/mcv_var_1.csv'

target = 'BMI'

sampler_data = pd.read_csv(
        'data/raw/MCV_VAR_RFC_FECHA_LLAVE.csv'
)

periods = [201609, 201610, 201611, 201612, 201701,
           201702, 201703, 201704, 201705, 201706,
           201707, 201708, 201709, 201710, 201711]
sample = pd.Series([False]*sampler_data.shape[0])
sample_exclude = sample
sample, sample_exclude = sampler(sampler_data, periods, sample, sample_exclude, month_size=700)
print(sample.sum())
print(sample_exclude.sum())

pp_data = pd.read_csv(
    data_preprocessed_file
)[sample]

np.random.seed(0)
rand_split = np.random.rand(len(pp_data))
train_list = rand_split < 0.8
test_list = rand_split >= 0.8

data_train = pp_data[train_list]
data_test = pp_data[test_list]

feat_names = data_train.columns[2:50]

print('----------------- Finish loading ---------------')

train_y = data_train.loc[:, target].values
train_X = data_train.loc[:, feat_names].values

test_y = data_test.loc[:, target].values
test_X = data_test.loc[:, feat_names].values

svclassifier = SVC(kernel='rbf', probability=True)
svclassifier.fit(train_X, train_y)

y_pred = svclassifier.predict(test_X)


print(confusion_matrix(test_y, y_pred))
print(confusion_matrix(train_y, svclassifier.predict(train_X)))

svclassifier._predict_proba(train_X)

print('finish!')


