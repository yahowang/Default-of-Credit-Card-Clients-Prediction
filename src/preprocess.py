# +
import pandas as pd
import numpy as np
from matplotlib import pylab as plt
import matplotlib
from collections import Counter
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")
import math
# %matplotlib inline

mixed_features = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
continuous =['LIMIT_BAL','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
categoricals = ["SEX", "EDUCATION", "MARRIAGE"]

def preprocess_train(X):
    global continuous
    global mixed_features
    global categoricals
    
    X_train = X.copy()
    X_train.reset_index(drop=True, inplace=True)

    # Standard Scaler
    sscaler = StandardScaler()
    X_train[continuous] = sscaler.fit_transform(X_train[continuous])

    # One Hot
    ohe = OneHotEncoder(sparse=False, handle_unknown = "ignore")
    temp = ohe.fit_transform(X_train[categoricals])
    for i in range(len(ohe.get_feature_names())):
        feature_name = ohe.get_feature_names()[i].split("_")
        feature_name = categoricals[int(feature_name[0][1])] + "_" + feature_name[1]    
        temp_i = pd.DataFrame(temp[:, i], columns = [feature_name])
        X_train = pd.concat([X_train, temp_i], axis = 1)
    for cat in categoricals:
        X_train = X_train.drop(cat, axis = 1)
    
    # Minmax
    mmfeatures = ["AGE"] + mixed_features
    minmax = MinMaxScaler()
    X_train[mmfeatures] = minmax.fit_transform(X_train[mmfeatures])
    
    return X_train, sscaler, ohe, minmax

def preprocess_other(X, sscaler, ohe, minmax):
    global continuous
    global mixed_features
    global categoricals

    X_other = X.copy()
    X_other.reset_index(drop=True, inplace=True)

    # Minmax
    mmfeatures = ["AGE"] + mixed_features
    X_other[mmfeatures] = minmax.transform(X_other[mmfeatures])
    
    # Standard Scaler
    X_other[continuous] = sscaler.transform(X_other[continuous])

    # One Hot
    temp = ohe.transform(X_other[categoricals])
    feature_names = ohe.get_feature_names(categoricals)
    for i in range(len(ohe.get_feature_names())):
        feature_name = feature_names[i]
        temp_i = pd.DataFrame(temp[:, i], columns = [feature_name])
        X_other = pd.concat([X_other, temp_i], axis = 1)  
    for cat in categoricals:
        X_other = X_other.drop(cat, axis = 1)

    return X_other

