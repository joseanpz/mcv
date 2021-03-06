from itertools import chain, zip_longest
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, accuracy_score


def pretty_table(matrix):
    matrix_aux = chain.from_iterable(
        zip_longest(
            *(x.splitlines() for x in y),
            fillvalue='')
        for y in [[str(e) for e in row] for row in matrix])

    s = [[str(e) for e in row] for row in matrix_aux]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))


def predict_with_threshold(probs, threshold):
    preds = []
    for prob in probs:
        if prob > threshold:
            preds.append(1)
        else:
            preds.append(0)
    return np.array(preds)


def val_func(pred_probs, dmat):
    thrs = [0.6, 0.63, 0.65, 0.68, 0.7, 0.72]
    ret = []
    for thr in thrs:
        preds = predict_with_threshold(pred_probs, thr)
        score = f1_score(preds, dmat.get_float_info('label'))
        ret.append(('f1_score_thr{}'.format(thr), score))
    return ret


def get_xgb_feat_importances(clf):
    if isinstance(clf, xgb.XGBModel):
        # clf has been created by calling
        # xgb.XGBClassifier.fit() or xgb.XGBRegressor().fit()
        fscore = clf.booster().get_fscore()
    else:
        # clf has been created by calling xgb.train.
        # Thus, clf is an instance of xgb.Booster.
        fscore = clf.get_fscore()

    feat_importances = []
    for ft, score in fscore.items():
        feat_importances.append({'Feature': ft, 'Importance': score})
    feat_importances = pd.DataFrame(feat_importances)
    feat_importances = feat_importances.sort_values(
        by='Importance', ascending=False).reset_index(drop=True)
    # Divide the importances by the sum of all importances
    # to get relative importances. By using relative importances
    # the sum of all importances will equal to 1, i.e.,
    # np.sum(feat_importances['importance']) == 1
    feat_importances['Importance'] /= feat_importances['Importance'].sum()
    # Print the most important features and their importances
    print(feat_importances.head())
    return feat_importances


def sampler(data, periods, sample, sample_exclude, month_size=500, seed=0):
    if periods:
        period = (data['FECHA'] == periods.pop()) & (~sample_exclude)
        local_universe = data[period].drop_duplicates('RFC')
        print(local_universe.shape)
        period_sample = local_universe.sample(month_size, random_state=seed)
        sample = pd.Series(data.index.isin(period_sample.index)) | sample
        sample_exclude = data['RFC'].isin(period_sample['RFC']) | sample_exclude
        return sampler(data, periods, sample, sample_exclude, month_size, seed)
    else:
        return sample, sample_exclude