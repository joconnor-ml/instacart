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
    "xgb": XGBClassifier(n_estimators=16, learning_rate=0.2, max_depth=2)
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
    try:
        y = pd.read_csv("../data/reorder_targets.csv", index_col=0,
                        dtype=np.int32)
    except Exception as e:
        y = generate_reorder_targets()
        print(y.head())
        y = pd.merge(X[["order_id", "product_id"]],
                     y[["order_id", "product_id", "reordered"]],
                     left_on=["order_id", "product_id"],
                     right_on=["order_id", "product_id"],
                     how="left").fillna(0)
        print(y.head())
        print(y.shape, y.sum())
        y.to_csv("../data/reorder_targets.csv")

    for name, model in models.items():
        with open("../models/reorder_model_{}.pkl".format(name), "wb") as f:
            pkl.dump(model.fit(X, y["reordered"]), f)

            
def predict_reorder_models():
    X = generate_reorder_features("train")
    preds = {}
    for name, model in models.items():
        with open("../models/reorder_model_{}.pkl".format(name), "rb") as f:
            model = pkl.load(f)
            preds[name] = model.predict_proba(X)[:, 1]
    return pd.DataFrame(preds, index=X.index)

            
def validate_reorder_models():
    print("Generating reorder_model features")
    X = generate_reorder_features("train")
    user_df = generate_user_features("train")
    
    # merge into feature dataframe
    X = pd.merge(X, user_df, left_on="user_id", right_index=True)    
    X["ratio"] = X["n_unique"] / X["n_total"]
    X["ratio2"] =  X["n_total"] / X["order_number"]
    X["day_chunk"] = X["order_hour_of_day"] // 8
    X["weekend"] = X["order_dow"] > 4
    X["weekly"] = (X["days_since_prior_order"] % 7) == 0
    X["same_day"] = X["days_since_prior_order"] == 0
    X["next_day"] = X["days_since_prior_order"] == 1
    X["monthly"] = X["days_since_prior_order"] == 30
    try:
        y = pd.read_csv("../data/reorder_targets.csv", index_col=0,
                        dtype=np.int32)
    except Exception as e:
        y = generate_reorder_targets()
        print(y.head())
        y = pd.merge(X[["order_id", "product_id"]],
                     y[["order_id", "product_id", "reordered"]],
                     left_on=["order_id", "product_id"],
                     right_on=["order_id", "product_id"],
                     how="left").fillna(0)
        print(y.head())
        print(y.shape, y.sum())
        y.to_csv("../data/reorder_targets.csv")

    preds = []
    names = []
    print("Modelling")
    for name, model in models.items():
        preds.append(cross_val_predict(model,
                                       X, y["reordered"],
                                       method="predict_proba")[:, 1])
        names.append(name)

    preds = pd.DataFrame(np.array(preds).T, columns=names, index=X.index)
    preds["mean"] = preds.mean(axis=1)
    preds.to_csv("reorder_preds_train.csv")
    def lloss(x):return log_loss(y["reordered"], x)
    def auc(x):return roc_auc_score(y["reordered"], x)
    print(preds.agg([lloss, auc]))



if __name__ == "__main__":
    train_reorder_models()
    validate_reorder_models()
    preds = predict_none_models()
    preds.to_csv("../data/reorder_preds_test.csv")
