import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from lassonet import LassoNetClassifier


import pandas as pd




with open('svm_pois_ds_10', 'rb') as file_1:
    svm_pois_ds_10_array = pickle.load(file_1)
# print(pois_ds_ridge_10_array.shape)

with open('svm_pois_ds_15', 'rb') as file_1:
    svm_pois_ds_15_array = pickle.load(file_1)

with open('svm_pois_ds_20', 'rb') as file_1:
    svm_pois_ds_20_array = pickle.load(file_1)

with open('svm_pois_ds_25', 'rb') as file_1:
    svm_pois_ds_25_array = pickle.load(file_1)

with open('svm_pois_ds_30', 'rb') as file_1:
    svm_pois_ds_30_array = pickle.load(file_1)

with open('svm_pois_ds_35', 'rb') as file_1:
    svm_pois_ds_35_array = pickle.load(file_1)

with open('svm_pois_ds_40', 'rb') as file_1:
    svm_pois_ds_40_array = pickle.load(file_1)

with open('svm_pois_ds_y_10', 'rb') as file_1:
    svm_pois_ds_y_10_array = pickle.load(file_1)
# print(pois_ds_ridge_10_array.shape)

with open('svm_pois_ds_y_15', 'rb') as file_1:
    svm_pois_ds_y_15_array = pickle.load(file_1)

with open('svm_pois_ds_y_20', 'rb') as file_1:
    svm_pois_ds_y_20_array = pickle.load(file_1)

with open('svm_pois_ds_y_25', 'rb') as file_1:
    svm_pois_ds_y_25_array = pickle.load(file_1)

with open('svm_pois_ds_y_30', 'rb') as file_1:
    svm_pois_ds_y_30_array = pickle.load(file_1)

with open('svm_pois_ds_y_35', 'rb') as file_1:
    svm_pois_ds_y_35_array = pickle.load(file_1)

with open('svm_pois_ds_y_40', 'rb') as file_1:
    svm_pois_ds_y_40_array = pickle.load(file_1)

with open('ridge_tr_X', 'rb') as file_1:
    tr_X_array_0 = pickle.load(file_1)

with open('ridge_tr_y', 'rb') as file_1:
    tr_y_array_0 = pickle.load(file_1)

with open('spambase_test_x', 'rb') as file_1:
    test_X_array = pickle.load(file_1)

with open('spambase_test_y', 'rb') as file_1:
    test_y_array = pickle.load(file_1)

with open('ridge_pois_ds_10', 'rb') as file_1:
    ridge_pois_ds_10_array = pickle.load(file_1)


with open('ridge_pois_ds_15', 'rb') as file_1:
    ridge_pois_ds_15_array = pickle.load(file_1)

with open('ridge_pois_ds_20', 'rb') as file_1:
    ridge_pois_ds_20_array = pickle.load(file_1)

with open('ridge_pois_ds_25', 'rb') as file_1:
    ridge_pois_ds_25_array = pickle.load(file_1)

with open('ridge_pois_ds_30', 'rb') as file_1:
    ridge_pois_ds_30_array = pickle.load(file_1)

with open('ridge_pois_ds_35', 'rb') as file_1:
    ridge_pois_ds_35_array = pickle.load(file_1)

with open('ridge_pois_ds_40', 'rb') as file_1:
    ridge_pois_ds_40_array = pickle.load(file_1)

with open('ridge_pois_ds_y_10', 'rb') as file_1:
    ridge_pois_ds_y_10_array = pickle.load(file_1)

with open('ridge_pois_ds_y_15', 'rb') as file_1:
    ridge_pois_ds_y_15_array = pickle.load(file_1)

with open('ridge_pois_ds_y_20', 'rb') as file_1:
    ridge_pois_ds_y_20_array = pickle.load(file_1)

with open('ridge_pois_ds_y_25', 'rb') as file_1:
    ridge_pois_ds_y_25_array = pickle.load(file_1)

with open('ridge_pois_ds_y_30', 'rb') as file_1:
    ridge_pois_ds_y_30_array = pickle.load(file_1)

with open('ridge_pois_ds_y_35', 'rb') as file_1:
    ridge_pois_ds_y_35_array = pickle.load(file_1)

with open('ridge_pois_ds_y_40', 'rb') as file_1:
    ridge_pois_ds_y_40_array = pickle.load(file_1)

with open('pois_ds_svm_adv_X_array', 'rb') as file_1:
    pois_ds_svm_adv_X_array = pickle.load(file_1)

with open('pois_ds_svm_adv_y_array', 'rb') as file_1:
    pois_ds_svm_adv_y_array = pickle.load(file_1)

with open('pois_ds_ridge_adv_X_array', 'rb') as file_1:
    pois_ds_ridge_adv_X_array = pickle.load(file_1)

with open('pois_ds_ridge_adv_y_array', 'rb') as file_1:
    pois_ds_ridge_adv_y_array = pickle.load(file_1)

with open('pois_ds_logistic_adv_X_array', 'rb') as file_1:
    pois_ds_logistic_adv_X_array = pickle.load(file_1)

with open('pois_ds_logistic_adv_y_array', 'rb') as file_1:
    pois_ds_logistic_adv_y_array = pickle.load(file_1)


# tr_X_array = np.append(tr_X_array_0, svm_pois_ds_40_array, axis=0)
# tr_y_array = np.append(tr_y_array_0, svm_pois_ds_y_40_array, axis=0)

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


model = LassoNetClassifier(eps_start=1e-3, lambda_start=1000)
path = model.path(X_train, y_train)

n_selected = []
accuracy = []

for save in path:
    model.load(save.state_dict)
    n_selected.append(save.selected.sum())
    y_pred = model.predict(X_test)
    accuracy.append(accuracy_score(y_test, y_pred))

fig = plt.figure(figsize=(9, 6))
plt.grid(True)
plt.plot(n_selected, accuracy, linestyle = "-.",label='clean')
plt.legend()
plt.xlabel("number of selected features")
plt.ylabel("classification accuracy")
plt.title("Classification accuracy")
plt.savefig("accuracy.png")




#加入对抗样本


#10%
# tr_X_array = np.append(X_train, svm_pois_ds_10_array, axis = 0)
#
#
# tr_y_array = np.append(y_train, svm_pois_ds_y_10_array, axis = 0)
#
#
# model = LassoNetClassifier(eps_start=1e-3, lambda_start=1000)
# path = model.path(tr_X_array, tr_y_array)
#
# n_selected = []
# accuracy = []
#
# for save in path:
#     model.load(save.state_dict)
#     n_selected.append(save.selected.sum())
#     y_pred = model.predict(X_test)
#     accuracy.append(accuracy_score(y_test, y_pred))
#
#
# plt.grid(True)
# plt.plot(n_selected, accuracy, linestyle = "-",label='10%')
# plt.legend()
#
#
# #15%
# tr_X_array = np.append(X_train, svm_pois_ds_15_array, axis = 0)
# tr_y_array = np.append(y_train, svm_pois_ds_y_15_array, axis = 0)
#
#
# model = LassoNetClassifier(eps_start=1e-3, lambda_start=1000)
# path = model.path(tr_X_array, tr_y_array)
#
# n_selected = []
# accuracy = []
#
# for save in path:
#     model.load(save.state_dict)
#     n_selected.append(save.selected.sum())
#     y_pred = model.predict(X_test)
#     accuracy.append(accuracy_score(y_test, y_pred))
#
# # fig = plt.figure(figsize=(9, 6))
# plt.grid(True)
# plt.plot(n_selected, accuracy, linestyle = "-",label='15%')
# plt.legend()
# plt.xlabel("number of selected features")
# plt.ylabel("classification accuracy")
# plt.title("Classification accuracy")
# plt.savefig("accuracy.png")

#20%-adv
# tr_X_array = np.append(X_train, ridge_pois_ds_20_array, axis = 0)
#
# # ridge_pois_ds_y_20_array_adv = []
#
# for i in range(len(ridge_pois_ds_y_20_array) >> 1):
#     if ridge_pois_ds_y_20_array[i] == 0:
#         ridge_pois_ds_y_20_array[i] = 1
#     else:
#         ridge_pois_ds_y_20_array[i] = 0
#
# tr_y_array = np.append(y_train, ridge_pois_ds_y_20_array, axis = 0)
#
# # print(svm_pois_ds_y_20_array)
#
# model = LassoNetClassifier(eps_start=1e-3, lambda_start=1000)
#
# path = model.path(tr_X_array, tr_y_array)
#
# n_selected = []
# accuracy = []
#
# for save in path:
#     model.load(save.state_dict)
#     n_selected.append(save.selected.sum())
#     y_pred = model.predict(X_test)
#     accuracy.append(accuracy_score(y_test, y_pred))
#
# # fig = plt.figure(figsize=(9, 6))
# plt.grid(True)
# plt.plot(n_selected, accuracy, linestyle = "--",label='20%-adv')
# plt.legend()
# plt.xlabel("number of selected features")
# plt.ylabel("classification accuracy")
# plt.title("Classification accuracy")
# plt.savefig("accuracy.png")
#
#
# with open('ridge_pois_ds_y_20', 'rb') as file_1:
#     ridge_pois_ds_y_20_array = pickle.load(file_1)


# #20%-poisoned
# tr_X_array = np.append(X_train, ridge_pois_ds_20_array, axis = 0)
# tr_y_array = np.append(y_train, ridge_pois_ds_y_20_array, axis = 0)
#
#
# model = LassoNetClassifier(eps_start=1e-3, lambda_start=1000)
# path = model.path(tr_X_array, tr_y_array)
#
# n_selected = []
# accuracy = []
#
# for save in path:
#     model.load(save.state_dict)
#     n_selected.append(save.selected.sum())
#     y_pred = model.predict(X_test)
#     accuracy.append(accuracy_score(y_test, y_pred))
#
# # fig = plt.figure(figsize=(9, 6))
# plt.grid(True)
# plt.plot(n_selected, accuracy, linestyle = ":",label='20%-poisoned')
# plt.legend()
# plt.xlabel("number of selected features")
# plt.ylabel("classification accuracy")
# plt.title("Classification accuracy")
# plt.savefig("accuracy.png")

# #25%-adv
# tr_X_array = np.append(X_train, ridge_pois_ds_25_array, axis = 0)
#
#
# for i in range(len(ridge_pois_ds_y_25_array) >> 1):
#     if ridge_pois_ds_y_25_array[i] == 0:
#         ridge_pois_ds_y_25_array[i] = 1
#     else:
#         ridge_pois_ds_y_25_array[i] = 0
#
# tr_y_array = np.append(y_train, ridge_pois_ds_y_25_array, axis = 0)
#
#
# model = LassoNetClassifier(eps_start=1e-3, lambda_start=1000)
#
# path = model.path(tr_X_array, tr_y_array)
#
# n_selected = []
# accuracy = []
#
# for save in path:
#     model.load(save.state_dict)
#     n_selected.append(save.selected.sum())
#     y_pred = model.predict(X_test)
#     accuracy.append(accuracy_score(y_test, y_pred))
#
# # fig = plt.figure(figsize=(9, 6))
# plt.grid(True)
# plt.plot(n_selected, accuracy, linestyle = "--",label='25%-adv')
# plt.legend()
# plt.xlabel("number of selected features")
# plt.ylabel("classification accuracy")
# plt.title("Classification accuracy")
# plt.savefig("accuracy.png")
#
#
# with open('ridge_pois_ds_y_25', 'rb') as file_1:
#     ridge_pois_ds_y_25_array = pickle.load(file_1)
#
#
# #20%-poisoned
# tr_X_array = np.append(X_train, ridge_pois_ds_25_array, axis = 0)
# tr_y_array = np.append(y_train, ridge_pois_ds_y_25_array, axis = 0)
#
#
# model = LassoNetClassifier(eps_start=1e-3, lambda_start=1000)
# path = model.path(tr_X_array, tr_y_array)
#
# n_selected = []
# accuracy = []
#
# for save in path:
#     model.load(save.state_dict)
#     n_selected.append(save.selected.sum())
#     y_pred = model.predict(X_test)
#     accuracy.append(accuracy_score(y_test, y_pred))
#
# # fig = plt.figure(figsize=(9, 6))
# plt.grid(True)
# plt.plot(n_selected, accuracy, linestyle = ":",label='25%-poisoned')
# plt.legend()
# # plt.xlabel("number of selected features")
# # plt.ylabel("classification accuracy")
# # plt.title("Classification accuracy")
# # plt.savefig("accuracy.png")

# #30%-poisoned
# tr_X_array = np.append(X_train, ridge_pois_ds_30_array, axis = 0)
# tr_y_array = np.append(y_train, ridge_pois_ds_y_30_array, axis = 0)
#
#
# model = LassoNetClassifier(eps_start=1e-3, lambda_start=1000)
# path = model.path(tr_X_array, tr_y_array)
#
# n_selected = []
# accuracy = []
#
# for save in path:
#     model.load(save.state_dict)
#     n_selected.append(save.selected.sum())
#     y_pred = model.predict(X_test)
#     accuracy.append(accuracy_score(y_test, y_pred))
#
# # fig = plt.figure(figsize=(9, 6))
# plt.grid(True)
# plt.plot(n_selected, accuracy, linestyle = ":",label='30%-poisoned')
# plt.legend()
#
# #30%-adv
# tr_X_array = np.append(X_train, ridge_pois_ds_30_array, axis = 0)
#
#
# for i in range(len(ridge_pois_ds_y_25_array) >> 1):
#     if ridge_pois_ds_y_30_array[i] == 0:
#         ridge_pois_ds_y_30_array[i] = 1
#     else:
#         ridge_pois_ds_y_30_array[i] = 0
#
# tr_y_array = np.append(y_train, ridge_pois_ds_y_30_array, axis = 0)
#
#
# model = LassoNetClassifier(eps_start=1e-3, lambda_start=1000)
#
# path = model.path(tr_X_array, tr_y_array)
#
# n_selected = []
# accuracy = []
#
# for save in path:
#     model.load(save.state_dict)
#     n_selected.append(save.selected.sum())
#     y_pred = model.predict(X_test)
#     accuracy.append(accuracy_score(y_test, y_pred))
#
# # fig = plt.figure(figsize=(9, 6))
# plt.grid(True)
# plt.plot(n_selected, accuracy, linestyle = "--",label='30%-adv')
# plt.legend()
# plt.xlabel("number of selected features")
# plt.ylabel("classification accuracy")
# plt.title("Classification accuracy")
# plt.savefig("accuracy.png")

# #30%-poisoned
# tr_X_array = np.append(X_train, ridge_pois_ds_30_array, axis = 0)
# tr_y_array = np.append(y_train, ridge_pois_ds_y_30_array, axis = 0)
#
#
# model = LassoNetClassifier(eps_start=1e-3, lambda_start=1000)
# path = model.path(tr_X_array, tr_y_array)
#
# n_selected = []
# accuracy = []
#
# for save in path:
#     model.load(save.state_dict)
#     n_selected.append(save.selected.sum())
#     y_pred = model.predict(X_test)
#     accuracy.append(accuracy_score(y_test, y_pred))
#
# # fig = plt.figure(figsize=(9, 6))
# plt.grid(True)
# plt.plot(n_selected, accuracy, linestyle = ":",label='30%-poisoned')
# plt.legend()
#
# #30%-adv
# tr_X_array = np.append(X_train, ridge_pois_ds_30_array, axis = 0)
#
#
# for i in range(len(ridge_pois_ds_y_30_array) >> 1):
#     if ridge_pois_ds_y_30_array[i] == 0:
#         ridge_pois_ds_y_30_array[i] = 1
#     else:
#         ridge_pois_ds_y_30_array[i] = 0
#
# tr_y_array = np.append(y_train, ridge_pois_ds_y_30_array, axis = 0)
#
#
# model = LassoNetClassifier(eps_start=1e-3, lambda_start=1000)
#
# path = model.path(tr_X_array, tr_y_array)
#
# n_selected = []
# accuracy = []
#
# for save in path:
#     model.load(save.state_dict)
#     n_selected.append(save.selected.sum())
#     y_pred = model.predict(X_test)
#     accuracy.append(accuracy_score(y_test, y_pred))
#
# # fig = plt.figure(figsize=(9, 6))
# plt.grid(True)
# plt.plot(n_selected, accuracy, linestyle = "--",label='30%-adv')
# plt.legend()
# plt.xlabel("number of selected features")
# plt.ylabel("classification accuracy")
# plt.title("Classification accuracy")
# plt.savefig("accuracy.png")

# #35%-poisoned
# tr_X_array = np.append(X_train, ridge_pois_ds_35_array, axis = 0)
# tr_y_array = np.append(y_train, ridge_pois_ds_y_35_array, axis = 0)
#
#
# model = LassoNetClassifier(eps_start=1e-3, lambda_start=1000)
# path = model.path(tr_X_array, tr_y_array)
#
# n_selected = []
# accuracy = []
#
# for save in path:
#     model.load(save.state_dict)
#     n_selected.append(save.selected.sum())
#     y_pred = model.predict(X_test)
#     accuracy.append(accuracy_score(y_test, y_pred))
#
# # fig = plt.figure(figsize=(9, 6))
# plt.grid(True)
# plt.plot(n_selected, accuracy, linestyle = ":",label='35%-poisoned')
# plt.legend()
#
# #35%-adv
# tr_X_array = np.append(X_train, ridge_pois_ds_35_array, axis = 0)
#
#
# for i in range(len(ridge_pois_ds_y_35_array) >> 1):
#     if ridge_pois_ds_y_35_array[i] == 0:
#         ridge_pois_ds_y_35_array[i] = 1
#     else:
#         ridge_pois_ds_y_35_array[i] = 0
#
# tr_y_array = np.append(y_train, ridge_pois_ds_y_35_array, axis = 0)
#
#
# model = LassoNetClassifier(eps_start=1e-3, lambda_start=1000)
#
# path = model.path(tr_X_array, tr_y_array)
#
# n_selected = []
# accuracy = []
#
# for save in path:
#     model.load(save.state_dict)
#     n_selected.append(save.selected.sum())
#     y_pred = model.predict(X_test)
#     accuracy.append(accuracy_score(y_test, y_pred))
#
# # fig = plt.figure(figsize=(9, 6))
# plt.grid(True)
# plt.plot(n_selected, accuracy, linestyle = "--",label='35%-adv')
# plt.legend()
# plt.xlabel("number of selected features")
# plt.ylabel("classification accuracy")
# plt.title("Classification accuracy")
# plt.savefig("accuracy.png")

# #40%-poisoned
# tr_X_array = np.append(X_train, ridge_pois_ds_40_array, axis = 0)
# tr_y_array = np.append(y_train, ridge_pois_ds_y_40_array, axis = 0)
#
#
# model = LassoNetClassifier(eps_start=1e-3, lambda_start=1000)
# path = model.path(tr_X_array, tr_y_array)
#
# n_selected = []
# accuracy = []
#
# for save in path:
#     model.load(save.state_dict)
#     n_selected.append(save.selected.sum())
#     y_pred = model.predict(X_test)
#     accuracy.append(accuracy_score(y_test, y_pred))
#
# # fig = plt.figure(figsize=(9, 6))
# plt.grid(True)
# plt.plot(n_selected, accuracy, linestyle = ":",label='40%-poisoned')
# plt.legend()
#
# #40%-adv
# tr_X_array = np.append(X_train, ridge_pois_ds_40_array, axis = 0)
#
#
# for i in range(len(ridge_pois_ds_y_40_array) >> 1):
#     if ridge_pois_ds_y_40_array[i] == 0:
#         ridge_pois_ds_y_40_array[i] = 1
#     else:
#         ridge_pois_ds_y_40_array[i] = 0
#
# tr_y_array = np.append(y_train, ridge_pois_ds_y_40_array, axis = 0)
#
#
# model = LassoNetClassifier(eps_start=1e-3, lambda_start=1000)
#
# path = model.path(tr_X_array, tr_y_array)
#
# n_selected = []
# accuracy = []
#
# for save in path:
#     model.load(save.state_dict)
#     n_selected.append(save.selected.sum())
#     y_pred = model.predict(X_test)
#     accuracy.append(accuracy_score(y_test, y_pred))
#
# # fig = plt.figure(figsize=(9, 6))
# plt.grid(True)
# plt.plot(n_selected, accuracy, linestyle = "--",label='40%-adv')
# plt.legend()
# plt.xlabel("number of selected features")
# plt.ylabel("classification accuracy")
# plt.title("Classification accuracy")
# plt.savefig("accuracy.png")


# #10%-poisoned
# tr_X_array = np.append(X_train, ridge_pois_ds_10_array, axis = 0)
# tr_y_array = np.append(y_train, ridge_pois_ds_y_10_array, axis = 0)
#
#
# model = LassoNetClassifier(eps_start=1e-3, lambda_start=1000)
# path = model.path(tr_X_array, tr_y_array)
#
# n_selected = []
# accuracy = []
#
# for save in path:
#     model.load(save.state_dict)
#     n_selected.append(save.selected.sum())
#     y_pred = model.predict(X_test)
#     accuracy.append(accuracy_score(y_test, y_pred))
#
# # fig = plt.figure(figsize=(9, 6))
# plt.grid(True)
# plt.plot(n_selected, accuracy, linestyle = ":",label='10%-poisoned')
# plt.legend()
#
# #10%-adv
# tr_X_array = np.append(X_train, ridge_pois_ds_10_array, axis = 0)
#
#
# for i in range(len(ridge_pois_ds_y_10_array) >> 1):
#     if ridge_pois_ds_y_10_array[i] == 0:
#         ridge_pois_ds_y_10_array[i] = 1
#     else:
#         ridge_pois_ds_y_10_array[i] = 0
#
# tr_y_array = np.append(y_train, ridge_pois_ds_y_10_array, axis = 0)
#
#
# model = LassoNetClassifier(eps_start=1e-3, lambda_start=1000)
#
# path = model.path(tr_X_array, tr_y_array)
#
# n_selected = []
# accuracy = []
#
# for save in path:
#     model.load(save.state_dict)
#     n_selected.append(save.selected.sum())
#     y_pred = model.predict(X_test)
#     accuracy.append(accuracy_score(y_test, y_pred))
#
# # fig = plt.figure(figsize=(9, 6))
# plt.grid(True)
# plt.plot(n_selected, accuracy, linestyle = "--",label='10%-adv')
# plt.legend()
# plt.xlabel("number of selected features")
# plt.ylabel("classification accuracy")
# plt.title("Classification accuracy")
# plt.savefig("accuracy.png")

# #15%-poisoned
# tr_X_array = np.append(X_train, ridge_pois_ds_15_array, axis = 0)
# tr_y_array = np.append(y_train, ridge_pois_ds_y_15_array, axis = 0)
#
#
# model = LassoNetClassifier(eps_start=1e-3, lambda_start=1000)
# path = model.path(tr_X_array, tr_y_array)
#
# n_selected = []
# accuracy = []
#
# for save in path:
#     model.load(save.state_dict)
#     n_selected.append(save.selected.sum())
#     y_pred = model.predict(X_test)
#     accuracy.append(accuracy_score(y_test, y_pred))
#
# # fig = plt.figure(figsize=(9, 6))
# plt.grid(True)
# plt.plot(n_selected, accuracy, linestyle = ":",label='15%-poisoned')
# plt.legend()
#
# #15%-adv
# tr_X_array = np.append(X_train, ridge_pois_ds_15_array, axis = 0)
#
#
# for i in range(len(ridge_pois_ds_y_15_array)):
#     if ridge_pois_ds_y_15_array[i] == 0:
#         ridge_pois_ds_y_15_array[i] = 1
#     else:
#         ridge_pois_ds_y_15_array[i] = 0
#
# tr_y_array = np.append(y_train, ridge_pois_ds_y_15_array, axis = 0)
#
#
# model = LassoNetClassifier(eps_start=1e-3, lambda_start=1000)
#
# path = model.path(tr_X_array, tr_y_array)
#
# n_selected = []
# accuracy = []
#
# for save in path:
#     model.load(save.state_dict)
#     n_selected.append(save.selected.sum())
#     y_pred = model.predict(X_test)
#     accuracy.append(accuracy_score(y_test, y_pred))
#
# # fig = plt.figure(figsize=(9, 6))
# plt.grid(True)
# plt.plot(n_selected, accuracy, linestyle = "--",label='15%-adv')
# plt.legend()
# plt.xlabel("number of selected features")
# plt.ylabel("classification accuracy")
# plt.title("Classification accuracy")
# plt.savefig("accuracy.png")

#10%-poisoned
tr_X_array = np.append(X_train, ridge_pois_ds_10_array, axis = 0)
tr_y_array = np.append(y_train, ridge_pois_ds_y_10_array, axis = 0)


model = LassoNetClassifier(eps_start=1e-3, lambda_start=1000)
path = model.path(tr_X_array, tr_y_array)

n_selected = []
accuracy = []

for save in path:
    model.load(save.state_dict)
    n_selected.append(save.selected.sum())
    y_pred = model.predict(X_test)
    accuracy.append(accuracy_score(y_test, y_pred))

# fig = plt.figure(figsize=(9, 6))
plt.grid(True)
plt.plot(n_selected, accuracy, linestyle = ":",label='10%-poisoned')
plt.legend()

#10%-adv
# tr_X_array = np.append(pois_ds_ridge_adv_X_array, ridge_pois_ds_20_array, axis = 0)

print(pois_ds_ridge_adv_y_array)

for i in range(len(pois_ds_ridge_adv_y_array)):
    if pois_ds_ridge_adv_y_array[i] == 0:
        pois_ds_ridge_adv_y_array[i] = 1
    else:
        pois_ds_ridge_adv_y_array[i] = 0

print(pois_ds_ridge_adv_y_array)

# tr_y_array = np.append(pois_ds_ridge_adv_y_array, ridge_pois_ds_y_20_array, axis = 0)
tr_X_array = pois_ds_ridge_adv_X_array
tr_y_array = pois_ds_ridge_adv_y_array


model = LassoNetClassifier(eps_start=1e-3, lambda_start=1000)

path = model.path(tr_X_array, tr_y_array)

n_selected = []
accuracy = []



for save in path:
    model.load(save.state_dict)
    n_selected.append(save.selected.sum())
    y_pred = model.predict(X_test)
    accuracy.append(accuracy_score(y_test, y_pred))

# fig = plt.figure(figsize=(9, 6))
plt.grid(True)
plt.plot(n_selected, accuracy, linestyle = "--",label='10%-adv')
plt.legend()
plt.xlabel("number of selected features")
plt.ylabel("classification accuracy")
plt.title("Classification accuracy")
plt.savefig("accuracy.png")



plt.show()



