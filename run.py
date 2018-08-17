import pandas as pd
import numpy as np
from sklearn import preprocessing as prp
import io
import boto3

f = io.BytesIO()
f_str = io.StringIO()

from sklearn.metrics import roc_curve, auc

#data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header = None)

# specify columns extracted from wbdc.names
#data.columns = ["id","diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
#                "compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean",
#                "radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se",
#                "concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst",
#                "perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst",
#                "concave points_worst","symmetry_worst","fractal_dimension_worst"]

# save the data
#data.to_csv("data.csv", sep=',', index=False)

# print the shape of the data file

data = pd.read_csv('titanic_test.csv', header=None)
data.to_csv(f_str, header=False)

s3_res = boto3.resource('s3')


print(data.shape)
#data = data.loc[:, ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare', 'sex', 'embarked', 'boat']]
# show the top few rows
print(data.head())

df = data.loc[:, ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']]

#labelencoder = prp.LabelEncoder()
#onehotencoder = prp.OneHotEncoder(categorical_features=['sex'])

# df['sex'] = labelencoder.fit_transform(df.loc[:, ['sex']])
# df['embarked'] = labelencoder.fit_transform(df.loc[:, ['embarked']])
# df['boat'] = labelencoder.fit_transform(df.loc[:, ['boat']])

# df = onehotencoder.fit_transform(df)
imputer = prp.Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(df.iloc[:, 1:])
df.loc[:, ['pclass', 'age', 'sibsp', 'parch', 'fare']] = imputer.transform(df.iloc[:, 1:])

df.loc[:, ['pclass', 'age', 'sibsp', 'parch', 'fare']] = prp.StandardScaler().fit_transform(df.drop('survived', axis=1))

rand_split = np.random.rand(len(df))
train_list = rand_split < 0.8
val_list = (rand_split >= 0.8) & (rand_split < 0.9)
test_list = rand_split >= 0.9

data_train = df[train_list]
data_val = df[val_list]
data_test = df[test_list]

train_y = data_train.iloc[:,0].as_matrix();
train_X = data_train.iloc[:,1:].as_matrix();

val_y = data_val.iloc[:,0].as_matrix();
val_X = data_val.iloc[:,1:].as_matrix();

test_y = data_test.iloc[:,0].as_matrix();
test_X = data_test.iloc[:,1:].as_matrix();

#data = data[['survided', 'pclass', 'sex', ]]
# describe the data object
print(data.describe())







# we will also summarize the categorical field diganosis
print(data.diagnosis.value_counts())

rand_split = np.random.rand(len(data))
train_list = rand_split < 0.8
val_list = (rand_split >= 0.8) & (rand_split < 0.9)
test_list = rand_split >= 0.9

data_train = data[train_list]
data_val = data[val_list]
data_test = data[test_list]

# al sumar 0 se convierten lo booleanos a enteros binarios 0, 1
train_y = ((data_train.iloc[:,1] == 'M') +0).as_matrix();
train_X = data_train.iloc[:,2:].as_matrix();

val_y = ((data_val.iloc[:,1] == 'M') +0).as_matrix();
val_X = data_val.iloc[:,2:].as_matrix();

test_y = ((data_test.iloc[:,1] == 'M') +0).as_matrix();
test_X = data_test.iloc[:,2:].as_matrix();

print('end!')