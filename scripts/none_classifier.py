import pickle as pkl
import numpy as np
import pandas as pd
import operator
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import cross_val_predict, cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer, StandardScaler, MinMaxScaler, PolynomialFeatures, OneHotEncoder
from sklearn.metrics import log_loss, roc_auc_score, f1_score
from xgboost import XGBClassifier
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from data_utils import *

models = {
    "linear": make_pipeline(StandardScaler(), LogisticRegression()),
    "xgb": XGBClassifier(n_estimators=16, learning_rate=0.5, max_depth=5)
}


def train_none_models():
    X = generate_none_features("train")
    y = generate_none_targets().loc[X.index]  # make sure ordering is correct
    for name, model in models.items():
        with open("../models/none_model_{}.pkl".format(name), "wb") as f:
            pkl.dump(model.fit(X, y), f)

            
def predict_none_models():
    X = generate_none_features("test")
    preds = {}
    for name, model in models.items():
        with open("../models/none_model_{}.pkl".format(name), "rb") as f:
            model = pkl.load(f)
            preds[name] = model.predict_proba(X)[:, 1]
    return pd.DataFrame(preds, index=X.index)


def validate_none_models():
    print("Generating none_model features")
    X = generate_none_features("train")
    print("Generating none_model targets")
    y = generate_none_targets().loc[X.index]  # make sure ordering is correct
    preds = []
    names = []

    print("Modelling")
    for name, model in models.items():
        preds.append(cross_val_predict(model,
                                       X, y,
                                       method="predict_proba")[:, 1])
        names.append(name)

    preds = pd.DataFrame(np.array(preds).T, columns=names, index=X.index)
    preds["mean"] = preds.mean(axis=1)
    preds.to_csv("../data/none_preds_train.csv")
    def ll(x):return log_loss(y, x)
    def auc(x):return roc_auc_score(y, x)
    print(preds.agg([ll, auc]))

    
if __name__ == "__main__":
    validate_none_models()
    train_none_models()
    preds = predict_none_models()
    preds.to_csv("../data/none_preds_test.csv")
