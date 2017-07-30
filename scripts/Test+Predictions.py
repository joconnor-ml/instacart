
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
test = orders[orders["eval_set"] == "train"]
test_users = test["user_id"].unique()
train = orders[(orders["eval_set"] == "prior") & (orders["user_id"].isin(test_users))]

truth = pd.read_csv('../data/order_products__train.csv', dtype=np.int32, index_col=0).groupby("order_id")["product_id"].apply(list)
truth.head()


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


with open("user_products.pkl", "wb") as f:
    pkl.dump(user_products, f)


# In[7]:


rows = []
for user, products in user_products.items():
    if user % 10000 == 0: print(user)
    if user > 60000: break
    product_counts = pd.Series(products.sum()).value_counts()
    total_orders = products.shape[0]
    for product, count in product_counts.iteritems():
        in_last_order1 = product in products.iloc[-1]
        in_last_order2 = product in products.iloc[-2]
        in_last_order3 = product in products.iloc[-3]
        rows.append([product, user, count, in_last_order1, in_last_order2, in_last_order3, total_orders]) #... etc


# In[8]:


df = pd.DataFrame(rows, columns=["product_id", "user_id", "product_count",
                                 "in_last_order1", "in_last_order2", "in_last_order3",
                                 "total_orders"])
df["order_frac"] = df["product_count"] / df["total_orders"]
df = pd.merge(df, test.drop(["eval_set"], axis=1), on="user_id")
df = pd.merge(df, pd.read_csv("../data/products.csv", index_col=0,
                              usecols=["product_id", "aisle_id", "department_id"]),
              how="left", left_on="product_id", right_index=True)
df.tail()


# In[9]:


for i in np.linspace(0.1, 0.5, 5):
    a = pd.DataFrame(df[df["order_frac"] >= i].groupby("user_id")["product_id"].apply(list))
    results = pd.merge(test, a, left_on="user_id", right_index=True).join(truth, on="order_id", rsuffix="_pred")
    print(i, f1_score(results["product_id"], results["product_id_pred"]))


# In[10]:


y = []
for i, row in df.iterrows():
    y.append(row["product_id"] in truth[row["order_id"]])


# In[11]:


p1 = cross_val_predict(make_pipeline(MinMaxScaler(), LogisticRegression()), df, y, method="predict_proba")
print(log_loss(y, p1), roc_auc_score(y, p1[:, 1]))


# In[23]:


p2 = cross_val_predict(XGBClassifier(n_estimators=8, learning_rate=0.4, max_depth=2),
                      df.drop(["pred", "y"], axis=1), y, method="predict_proba")
print(log_loss(y, p2), roc_auc_score(y, p2[:, 1]))
print(log_loss(y, (p1+p2)/2), roc_auc_score(y, ((p1+p2)/2)[:, 1]))


# In[13]:


p = df["order_frac"]
print(log_loss(y, p), roc_auc_score(y, p))


# In[14]:


model = make_pipeline(MinMaxScaler(), LogisticRegression()).fit(df, y)
with open("linear.pkl", "wb")as f:
    pkl.dump(model, f)
model = make_pipeline(XGBClassifier(n_estimators=8, learning_rate=0.4)).fit(df, y)
with open("xgb.pkl", "wb")as f:
    pkl.dump(model, f)


# In[15]:


df["pred"] = p2[:,1]


# In[16]:


points = []
for i in np.linspace(0.05, 0.2, 16):
    a = pd.DataFrame(df[df["pred"] >= i].groupby("user_id")["product_id"].apply(list))
    results = pd.merge(test, a, left_on="user_id", right_index=True).join(truth, on="order_id", rsuffix="_pred")
    score = f1_score(results["product_id"], results["product_id_pred"])
    print(i, score)
    points.append(score)


# In[17]:


df["y"] = y


# In[18]:


df


# In[20]:


get_ipython().magic('matplotlib inline')
pd.Series(points, index=np.linspace(0.05, 0.2, 16)).plot()


# In[ ]:




