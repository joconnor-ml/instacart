import pickle as pkl
import numpy as np
import pandas as pd
import operator
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import cross_val_predict, cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer, StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.metrics import log_loss, roc_auc_score, f1_score
from xgboost import XGBClassifier
from data_utils import *


models = {
    "linear": make_pipeline(StandardScaler(), LogisticRegression()),
    "xgb": XGBClassifier(n_estimators=16, learning_rate=0.5, max_depth=5)
}


def f1_score_single(y_true, y_pred):
    try:
        y_true = set(list(y_true))
        y_pred = set(list(y_pred))
        cross_size = len(y_true & y_pred)
        if cross_size == 0: return 0.
        p = 1. * cross_size / len(y_pred)
        r = 1. * cross_size / len(y_true)
        return 2 * p * r / (p + r)
    except:
        return 0
    

def f1_score(y_true, y_pred):
    return np.mean([f1_score_single(x, y) for x, y in zip(y_true, y_pred)])


def train_reorder_models():
    X = generate_reorder_features("train")
    print(X.head())
    targets = generate_reorder_targets()
    print(targets.head())
    y = []
    for i, row in X.iterrows():
        y.append(row["product_id"] in targets["order_id"])

    for name, model in models.items():
        with open("../models/reorder_model_{}.pkl".format(name), "wb") as f:
            pkl.dump(model.fit(X, y), f)

            
def validate_reorder_models():
    print("Generating reorder_model features")
    X = generate_reorder_features("train")
    print("Generating reorder_model targets")
    targets = generate_reorder_targets()
    y = []
    for i, row in X.iterrows():
        y.append(row["product_id"] in truth[row["order_id"]])

    preds = []
    names = []
    print("Modelling")
    for name, model in models.items():
        preds.append(cross_val_predict(model,
                                       X, y,
                                       method="predict_proba")[:, 1])
        names.append(name)

    preds = pd.DataFrame(np.array(preds).T, columns=names)
    preds["mean"] = preds.mean(axis=1)
    def log_loss(x):return log_loss(y, x),
    def roc_auc(x):return roc_auc_score(y, x),
    print(preds.agg([log_loss, roc_auc]))



if __name__ == "__main__":
    train_reorder_models()
    validate_reorder_models()
