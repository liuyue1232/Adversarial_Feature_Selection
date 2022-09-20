import pickle
import numpy as np

with open('logistic_pois_ds_10', 'rb') as file_1:
    logistic_pois_ds_10_array = pickle.load(file_1)
# print(pois_ds_ridge_10_array.shape)

with open('logistic_pois_ds_15', 'rb') as file_1:
    logistic_pois_ds_15_array = pickle.load(file_1)

with open('logistic_pois_ds_20', 'rb') as file_1:
    logistic_pois_ds_20_array = pickle.load(file_1)

with open('logistic_pois_ds_25', 'rb') as file_1:
    logistic_pois_ds_25_array = pickle.load(file_1)

with open('logistic_pois_ds_30', 'rb') as file_1:
    logistic_pois_ds_30_array = pickle.load(file_1)

with open('logistic_pois_ds_35', 'rb') as file_1:
    logistic_pois_ds_35_array = pickle.load(file_1)

with open('logistic_pois_ds_40', 'rb') as file_1:
    logistic_pois_ds_40_array = pickle.load(file_1)

with open('logistic_pois_ds_y_10', 'rb') as file_1:
    logistic_pois_ds_y_10_array = pickle.load(file_1)
# print(pois_ds_ridge_10_array.shape)

with open('logistic_pois_ds_y_15', 'rb') as file_1:
    logistic_pois_ds_y_15_array = pickle.load(file_1)

with open('logistic_pois_ds_y_20', 'rb') as file_1:
    logistic_pois_ds_y_20_array = pickle.load(file_1)

with open('logistic_pois_ds_y_25', 'rb') as file_1:
    logistic_pois_ds_y_25_array = pickle.load(file_1)

with open('logistic_pois_ds_y_30', 'rb') as file_1:
    logistic_pois_ds_y_30_array = pickle.load(file_1)

with open('logistic_pois_ds_y_35', 'rb') as file_1:
    logistic_pois_ds_y_35_array = pickle.load(file_1)

with open('logistic_pois_ds_y_40', 'rb') as file_1:
    logistic_pois_ds_y_40_array = pickle.load(file_1)

with open('ridge_tr_X', 'rb') as file_1:
    tr_X_array_0 = pickle.load(file_1)

with open('ridge_tr_y', 'rb') as file_1:
    tr_y_array_0 = pickle.load(file_1)

with open('spambase_test_x', 'rb') as file_1:
    test_X_array = pickle.load(file_1)

with open('spambase_test_y', 'rb') as file_1:
    test_y_array = pickle.load(file_1)


#!/usr/bin/env python
# coding: utf-8

"""
Lassonet Demo Notebook - PyTorch

This notebook illustrates the Lassonet method for
feature selection on a classification task.
We will run Lassonet over [the Mice Dataset](https://archive.ics.uci.edu/ml/datasets/Mice%20Protein%20Expression).
This dataset consists of protein expression levels measured in the cortex of normal and trisomic mice who had been exposed to different experimental conditions. Each feature is the expression level of one protein.
"""
# First we import a few necessary packages


import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from lassonet import LassoNetClassifier


import pandas as pd


def load_mice():
    df = pd.read_csv("./spambase/spambase.data")
    # y = list(df[df.columns[57]].itertuples(False))
    y = list(df[df.columns[57]])
    classes = {lbl: i for i, lbl in enumerate(sorted(set(y)))}
    y = np.array([classes[lbl] for lbl in y])
    feats = df.columns[0:57]
    X = df[feats].fillna(df.groupby(y)[feats].transform("mean")).values
    X = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)
    return X, y


X, y = load_mice()


X_train, X_test, y_train, y_test = train_test_split(X, y)

# 干净样本
model = LassoNetClassifier(eps_start=1e-3, lambda_start=1000)
path = model.path(X_train, y_train)

n_selected = []
accuracy = []
topk = 10

print("clean:")

for save in path:
    model.load(save.state_dict)
    print(save.selected.tolist())
    n_selected.append(save.selected.sum())
    y_pred = model.predict(X_test)
    skip_weight = save.state_dict['skip.weight']
    inds = np.argsort(skip_weight[0])[-topk:]
    print("indexs={}".format(inds.tolist()))
    inds = np.argsort(skip_weight[1])[-topk:]
    print("indexs={}".format(inds.tolist()))
    accuracy.append(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))

fig = plt.figure(figsize=(9, 6))
plt.grid(True)
plt.plot(n_selected, accuracy, linestyle = ":",label='clean')
plt.legend()
plt.xlabel("number of selected features")
plt.ylabel("classification accuracy")
plt.title("Classification accuracy")
plt.savefig("accuracy.png")



#加入对抗样本


#10%
tr_X_array = np.append(X_train, logistic_pois_ds_10_array, axis = 0)
tr_y_array = np.append(y_train, logistic_pois_ds_y_10_array, axis = 0)


model = LassoNetClassifier(eps_start=1e-3, lambda_start=1000)
path = model.path(tr_X_array, tr_y_array)

n_selected = []
accuracy = []

print("10%")

for save in path:
    model.load(save.state_dict)
    print(save.selected.tolist())
    n_selected.append(save.selected.sum())
    y_pred = model.predict(X_test)
    skip_weight = save.state_dict['skip.weight']
    inds = np.argsort(skip_weight[0])[-topk:]
    print("indexs={}".format(inds.tolist()))
    inds = np.argsort(skip_weight[1])[-topk:]
    print("indexs={}".format(inds.tolist()))
    accuracy.append(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))

plt.grid(True)
plt.plot(n_selected, accuracy, label = "10%",linestyle = "-")
plt.legend()


# #15%
# tr_X_array = np.append(X_train, ridge_pois_ds_15_array, axis = 0)
# tr_y_array = np.append(y_train, ridge_pois_ds_y_15_array, axis = 0)


# model = LassoNetClassifier(eps_start=1e-3, lambda_start=1000)
# path = model.path(tr_X_array, tr_y_array)

# n_selected = []
# accuracy = []

# for save in path:
#     model.load(save.state_dict)
#     n_selected.append(save.selected.sum())
#     y_pred = model.predict(X_test)
#     accuracy.append(accuracy_score(y_test, y_pred))

# # fig = plt.figure(figsize=(9, 6))
# plt.grid(True)
# plt.plot(n_selected, accuracy, linestyle = "--",label = "15%")
# plt.legend()
# # plt.xlabel("number of selected features")
# # plt.ylabel("classification accuracy")
# # plt.title("Classification accuracy")
# # plt.savefig("accuracy.png")


#20%
tr_X_array = np.append(X_train, logistic_pois_ds_20_array, axis = 0)
tr_y_array = np.append(y_train, logistic_pois_ds_y_20_array, axis = 0)


model = LassoNetClassifier(eps_start=1e-3, lambda_start=1000)
path = model.path(tr_X_array, tr_y_array)

n_selected = []
accuracy = []

print("20%")

for save in path:
    model.load(save.state_dict)
    print(save.selected.tolist())
    n_selected.append(save.selected.sum())
    y_pred = model.predict(X_test)
    skip_weight = save.state_dict['skip.weight']
    inds = np.argsort(skip_weight[0])[-topk:]
    print("indexs={}".format(inds.tolist()))
    inds = np.argsort(skip_weight[1])[-topk:]
    print("indexs={}".format(inds.tolist()))
    accuracy.append(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))

# fig = plt.figure(figsize=(9, 6))
plt.grid(True)
plt.plot(n_selected, accuracy, linestyle = "--",label = "20%")
plt.legend()
# plt.xlabel("number of selected features")
# plt.ylabel("classification accuracy")
# plt.title("Classification accuracy")
# plt.savefig("accuracy.png")

# #25%
# tr_X_array = np.append(X_train, ridge_pois_ds_25_array, axis = 0)
# tr_y_array = np.append(y_train, ridge_pois_ds_y_25_array, axis = 0)


# model = LassoNetClassifier(eps_start=1e-3, lambda_start=1000)
# path = model.path(tr_X_array, tr_y_array)

# n_selected = []
# accuracy = []

# for save in path:
#     model.load(save.state_dict)
#     n_selected.append(save.selected.sum())
#     y_pred = model.predict(X_test)
#     accuracy.append(accuracy_score(y_test, y_pred))

# # fig = plt.figure(figsize=(9, 6))
# plt.grid(True)
# plt.plot(n_selected, accuracy, linestyle = "None",label = "25%")
# plt.legend()
# # plt.xlabel("number of selected features")
# # plt.ylabel("classification accuracy")
# # plt.title("Classification accuracy")
# # plt.savefig("accuracy.png")

#30%
tr_X_array = np.append(X_train, logistic_pois_ds_30_array, axis = 0)
tr_y_array = np.append(y_train, logistic_pois_ds_y_30_array, axis = 0)


model = LassoNetClassifier(eps_start=1e-3, lambda_start=1000)
path = model.path(tr_X_array, tr_y_array)

n_selected = []
accuracy = []


print("30%")

for save in path:
    model.load(save.state_dict)
    print(save.selected.tolist())
    n_selected.append(save.selected.sum())
    y_pred = model.predict(X_test)
    skip_weight = save.state_dict['skip.weight']
    inds = np.argsort(skip_weight[0])[-topk:]
    print("indexs={}".format(inds.tolist()))
    inds = np.argsort(skip_weight[1])[-topk:]
    print("indexs={}".format(inds.tolist()))
    accuracy.append(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))

# fig = plt.figure(figsize=(9, 6))
plt.grid(True)
plt.plot(n_selected, accuracy, linestyle = "solid",label = "30%")
plt.legend()
# plt.xlabel("number of selected features")
# plt.ylabel("classification accuracy")
# plt.title("Classification accuracy")
# plt.savefig("accuracy.png")

# #35%
# tr_X_array = np.append(X_train, ridge_pois_ds_35_array, axis = 0)
# tr_y_array = np.append(y_train, ridge_pois_ds_y_35_array, axis = 0)


# model = LassoNetClassifier(eps_start=1e-3, lambda_start=1000)
# path = model.path(tr_X_array, tr_y_array)

# n_selected = []
# accuracy = []

# for save in path:
#     model.load(save.state_dict)
#     n_selected.append(save.selected.sum())
#     y_pred = model.predict(X_test)
#     accuracy.append(accuracy_score(y_test, y_pred))

# # fig = plt.figure(figsize=(9, 6))
# plt.grid(True)
# plt.plot(n_selected, accuracy, linestyle = ":",label = "35%")
# plt.legend()
# # plt.xlabel("number of selected features")
# # plt.ylabel("classification accuracy")
# # plt.title("Classification accuracy")
# # plt.savefig("accuracy.png")


#40%
tr_X_array = np.append(X_train, logistic_pois_ds_40_array, axis = 0)
tr_y_array = np.append(y_train, logistic_pois_ds_y_40_array, axis = 0)


model = LassoNetClassifier(eps_start=1e-3, lambda_start=1000)
path = model.path(tr_X_array, tr_y_array)

n_selected = []
accuracy = []
skip_weight = np.array([])
topk = 10

print("40%")

for save in path:
    model.load(save.state_dict)
    print(save.selected.tolist())
    n_selected.append(save.selected.sum())
    y_pred = model.predict(X_test)
    skip_weight = save.state_dict['skip.weight'].numpy()
    inds = np.argsort(skip_weight[0])[-topk:]
    print("indexs={}".format(inds.tolist()))
    inds = np.argsort(skip_weight[1])[-topk:]
    print("indexs={}".format(inds.tolist()))
    accuracy.append(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))

# fig = plt.figure(figsize=(9, 6))
plt.grid(True)
plt.plot(n_selected, accuracy, linestyle = "-.", label = "40%")
plt.legend()
# plt.xlabel("number of selected features")
# plt.ylabel("classification accuracy")
# plt.title("Classification accuracy")
# plt.savefig("accuracy.png")

plt.show()