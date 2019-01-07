from importlib import reload
import pandas as pd
import numpy as np
import pickle
import io
import boto3

import xgboost as xgb


input_file_name = 'MCV_VARIABLES_201806_201810'

data = pd.read_csv(f'inputs/{input_file_name}.csv', sep='|')
data_copy = data.copy()

imputer = pickle.load(open("pickles/model3_1_imputer.data", "rb"))
scaler = pickle.load(open("pickles/model3_1_scaler.data", "rb"))
booster = pickle.load(open('pickles/model3_1_booster.data', 'rb'))
model_transform_features = pickle.load(open('pickles/model3_1_transform_features.data', 'rb'))
model_features = pickle.load(open('pickles/model3_1_features.data', 'rb'))

data.loc[:, model_transform_features] = imputer.transform(data.loc[:,model_transform_features].values)
data.loc[: ,model_transform_features] = scaler.transform(data.loc[:,model_transform_features].values)

X_eval = data[model_features].values
deval = xgb.DMatrix(
    X_eval,
    feature_names=model_features
)

data_copy.loc[:, 'PRED'] = pd.Series(booster.predict(deval)).reset_index(drop=True)
data_copy.to_csv(f'outputs/{input_file_name}_output.csv', index=False)
