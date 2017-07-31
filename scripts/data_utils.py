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
from collections import defaultdict

dtype={"order_id": np.int32,
       "user_id": np.int32,
       "eval_set": "category",
       "order_number": np.int32,
       "order_dow": np.int32,
       "order_hour_of_day": np.int32,
       "days_since_prior_order": np.float32}


def generate_order_features(eval_set):
    try:
        test = pd.read_csv("../data/order_features_{}.csv".format(eval_set),
                           index_col=0)
    except Exception as e:
        print(e)
        # get all the orders we're interested in
        orders = pd.read_csv('../data/orders.csv',
                             dtype=dtype,
                             index_col=0)
        test = orders[orders["eval_set"] == eval_set].copy()
        del orders
        test.to_csv("../data/order_features_{}.csv".format(eval_set))
        
    test["dummy"] = 1  # for benchmarking
    return test


def generate_user_products(eval_set):
    try:
        with open("../data/user_products_{}.pkl".format(eval_set), "rb") as f:
            user_products = pkl.load(f)
    except Exception as e:
        print(e)
        # get all the orders we're interested in
        orders = pd.read_csv('../data/orders.csv',
                             dtype=dtype)
        test = orders[orders["eval_set"] == eval_set].copy()
        test_users = test["user_id"].unique()
        train = orders[(orders["eval_set"] == "prior") &
                       (orders["user_id"].isin(test_users))]
        del orders
    
        user_orders = train.groupby("user_id")["order_id"].apply(list)

        train_products = pd.read_csv('../data/order_products__prior.csv',
                                     dtype=np.int32, index_col=0)
        train_products = train_products.loc[train["order_id"], "product_id"]\
                                       .groupby("order_id").apply(list)

        user_products = defaultdict(list)
        for user, order_ids in user_orders.iteritems():
            if user % 10000 == 0: print(user)
            products = train_products.loc[order_ids]
            user_products[user] = products

        with open("../data/user_products_{}.pkl".format(eval_set), "wb") as f:
            pkl.dump(user_products, f)

    return user_products


def generate_user_features(eval_set):
    try:
        user_df = pd.read_csv("../data/user_df_{}.csv".format(eval_set), index_col=0)
    except Exception as e:
        print(e)
        user_products = generate_user_products(eval_set)

        users = []
        bag_all_products = []
        bag_last_products = []
        n_unique = []
        n_total = []
    
        for user, products in user_products.items():
            if user % 10000 == 0: print(user)
            all_products = pd.Series(products.sum())
            n_unique.append(all_products.nunique())
            n_total.append(all_products.shape[0])
            bag_all_products.append(all_products)
            bag_last_products.append(pd.Series(products.iloc[-1]))
            users.append(user)

        rows, cols = [], []
        for i, row in enumerate(bag_all_products):
            for prod in row:
                rows.append(i)
                cols.append(prod)
        data = [1]*len(rows)

        bag_all_products = csr_matrix((data, (rows, cols)))
        svd_model = TruncatedSVD(n_iter=5, n_components=10)
        svd_all_products = svd_model.fit_transform(bag_all_products)
        del bag_all_products

        rows, cols = [], []
        for i, row in enumerate(bag_last_products):
            for prod in row:
                rows.append(i)
                cols.append(prod)
        data = [1]*len(rows)

        bag_last_products = csr_matrix((data, (rows, cols)))
        svd_last_products = svd_model.fit_transform(bag_last_products)
        del bag_last_products

        user_df = pd.DataFrame({
            "n_unique":n_unique,
            "n_total":n_total,
        }, index=users)
        user_df = pd.concat([
            user_df,
            pd.DataFrame(svd_last_products, index=users).add_prefix("svd_last_"), 
            pd.DataFrame(svd_all_products, index=users).add_prefix("svd_all_")
        ], axis=1)
        del user_products
    
        user_df.to_csv("../data/user_df_{}.csv".format(eval_set))
    return user_df


def generate_reorder_features(eval_set):
    try:
        df = pd.read_csv("../data/reorder_features_{}.csv".format(eval_set), index_col=0, dtype=np.int32)
    except Exception as e:
        print(e)
        user_products = generate_user_products(eval_set)
        test = generate_order_features(eval_set)
        test["order_id"] = test.index
        
        rows = []
        for user, products in user_products.items():
            if user % 10000 == 0: print(user)
            product_counts = pd.Series(products.sum()).value_counts()
            total_orders = products.shape[0]
            for product, count in product_counts.iteritems():
                in_last_order1 = product in products.iloc[-1]
                in_last_order2 = product in products.iloc[-2]
                in_last_order3 = product in products.iloc[-3]
                rows.append([product, user, count,
                             in_last_order1, in_last_order2, in_last_order3,
                             total_orders]) #... etc

        df = pd.DataFrame(rows, columns=["product_id", "user_id",
                                         "product_count",
                                         "in_last_order1", "in_last_order2",
                                         "in_last_order3", "total_orders"])
        df = pd.merge(df, test.drop(["eval_set"], axis=1), on="user_id")
        df = pd.merge(df, pd.read_csv("../data/products.csv", index_col=0,
                                      usecols=["product_id", "aisle_id",
                                               "department_id"]),
                      how="left", left_on="product_id", right_index=True)
        df.to_csv("../data/reorder_features_{}.csv".format(eval_set))
    
    df["order_frac"] = df["product_count"] / df["total_orders"]
    return df

def generate_reorder_targets():
    x = pd.read_csv('../data/order_products__train.csv', dtype=np.int32)
    return x[x["reordered"]==True]#x.groupby("order_id")["product_id"].apply(list)


def generate_none_features(eval_set):
    try:
        test = pd.read_csv("../data/none_features_{}.csv".format(eval_set), index_col=0)
    except Exception as e:
        print(e)
        test = generate_order_features(eval_set)
        user_df = generate_user_features(eval_set)
    
        # merge into feature dataframe
        test = pd.merge(test, user_df, left_on="user_id", right_index=True)    
        test.to_csv("../data/none_features_{}.csv".format(eval_set))
    
    X = test.drop(["user_id", "eval_set", "dummy"], axis=1)
    X["ratio"] = X["n_unique"] / X["n_total"]
    X["ratio2"] =  X["n_total"] / X["order_number"]
    X["day_chunk"] = X["order_hour_of_day"] // 8
    X["weekend"] = X["order_dow"] > 4
    X["weekly"] = (X["days_since_prior_order"] % 7) == 0
    X["same_day"] = X["days_since_prior_order"] == 0
    X["next_day"] = X["days_since_prior_order"] == 1
    X["monthly"] = X["days_since_prior_order"] == 30
    return X


def generate_none_targets():
    # count the number of reordered items for each basket:
    order_size = pd.read_csv('../data/order_products__train.csv',
                             dtype=np.int32, index_col=0)["reordered"]\
                   .groupby(level=0).sum()
    return (order_size == 0)
