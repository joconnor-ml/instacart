
# coding: utf-8

# In[1]:


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


# In[2]:


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


# In[3]:


orders = pd.read_csv('../data/orders.csv', dtype={"order_id": np.int32, "user_id": np.int32,
                                                  "eval_set": "category", "order_number": np.int32,
                                                  "order_dow": np.int32, "order_hour_of_day": np.int32,
                                                  "days_since_prior_order": np.float32})
test = orders[orders["eval_set"] == "test"]
test_users = test["user_id"].unique()
train = orders[(orders["eval_set"] == "prior") & (orders["user_id"].isin(test_users))]


# In[4]:


user_orders = train.groupby("user_id")["order_id"].apply(list)


# In[5]:


train_products = pd.read_csv('../data/order_products__prior.csv', dtype=np.int32, index_col=0)
train_products = train_products.loc[train["order_id"], "product_id"].groupby("order_id").apply(list)


# In[6]:


from collections import defaultdict
user_products = defaultdict(list)
for user, order_ids in user_orders.iteritems():
    if user % 10000 == 0: print(user)
    products = train_products.loc[order_ids]
    user_products[user] = products


# In[7]:


user_products[3]


# In[15]:


rows = []
for i, (user, products) in enumerate(user_products.items()):
    if i % 5000 == 0: print(i)
    product_counts = pd.Series(products.sum()).value_counts()
    total_orders = products.shape[0]
    for product, count in product_counts.iteritems():
        in_last_order1 = product in products.iloc[-1]
        in_last_order2 = product in products.iloc[-2]
        in_last_order3 = product in products.iloc[-3]
        rows.append([product, user, count, in_last_order1, in_last_order2, in_last_order3, total_orders]) #... etc


# In[22]:


df = pd.DataFrame(rows, columns=["product_id", "user_id", "product_count",
                                 "in_last_order1", "in_last_order2", "in_last_order3",
                                 "total_orders"])
df["order_frac"] = df["product_count"] / df["total_orders"]
df = pd.merge(df, test.drop(["eval_set"], axis=1), on="user_id")
df = pd.merge(df, pd.read_csv("../data/products.csv", index_col=0,
                              usecols=["product_id", "aisle_id", "department_id"]),
              how="left", left_on="product_id", right_index=True)
print(df.shape)
df.head()


# In[24]:


import pickle
with open("xgb.pkl", "rb") as f:
    model = pickle.load(f)
df["pred"] = 0
df["pred"].iloc[:100000] = model.predict_proba(df.iloc[:100000].drop("pred", axis=1))[:, 1]
df["pred"].iloc[100000:] = model.predict_proba(df.iloc[100000:].drop("pred", axis=1))[:, 1]
sub = df[df["pred"] > 0.13].groupby("order_id")["product_id"].apply(lambda x: " ".join(str(a) for a in x))
samp = pd.read_csv("../data/sample_submission.csv", index_col=0)
sub.loc[samp.index].fillna("None").to_csv("joe_xgb2.csv")


# In[ ]:




