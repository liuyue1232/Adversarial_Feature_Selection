import pandas as pd
import numpy as np
from secml.array import CArray
from secml.data import CDataset
from sklearn.model_selection import train_test_split
from secml.ml.features import CNormalizerMinMax
import matplotlib.pyplot as plt
import pickle
#导入数据

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




df = pd.read_csv('./spambase/spambase.data',header=None)
random_state = 999


df_tr_val = df[:4000] #训练集+验证集
data_tr_val = df_tr_val.drop([57],axis=1)
data_array = data_tr_val.values
target_tr_val = df_tr_val[57]



df_test = df[4000:]#测试集
data_test = df_test.drop([57],axis=1)
data_test_array = data_test.values
target_test = df_test[57]
test_x = data_test_array

for i in range(len(pois_ds_svm_adv_y_array)):
    if pois_ds_svm_adv_y_array[i] == 0:
        pois_ds_svm_adv_y_array[i] = 1
    else:
        pois_ds_svm_adv_y_array[i] = 0



#训练集与验证集的划分
train_x, valid_x, train_y, valid_y = train_test_split(data_array, target_tr_val, test_size=0.25, shuffle=True)
train_y = CArray(pois_ds_svm_adv_y_array)
train_x = CArray(pois_ds_svm_adv_X_array)
valid_x = CArray(valid_x)
valid_y = CArray(valid_y)
test_x = CArray(data_test_array)
test_y = CArray(target_test)



n_tr = 3000  # Number of training set samples
n_val = 1000  # Number of validation set samples
n_ts = 601  # Number of test set samples

# Normalize the data
nmz = CNormalizerMinMax()
tr_X = nmz.fit_transform(train_x)
val_X = nmz.transform(valid_x)
ts_X = nmz.transform(test_x)




# Metric to use for training and performance evaluation
from secml.ml.peval.metrics import CMetricAccuracy
metric = CMetricAccuracy()


# Creation of the multiclass classifier
from secml.ml.classifiers import CClassifierRidge
from secml.ml.classifiers import CClassifierSVM
from secml.ml.classifiers import CClassifierLogistic
from secml.ml.classifiers import CClassifierRandomForest
from secml.ml.kernels import CKernelRBF
clf = CClassifierSVM(kernel=CKernelRBF(gamma=10), C=1)
# clf = CClassifierRidge()
# clf = CClassifierLogistic()
# clf = CClassifierRandomForest()
# We can now fit the classifier
clf.fit(tr_X, train_y)
print("Training of classifier complete!")


# Compute predictions on a test set
y_pred = clf.predict(ts_X)

#设置CDataset格式
tr = CDataset(tr_X,train_y)
val = CDataset(val_X, valid_y)
ts = CDataset(ts_X, test_y)


#划分训练集为tr1和tr2

n_tr = 1500

tr_X_1 = tr_X.tondarray()[:n_tr]
tr_X_2 = tr_X.tondarray()[n_tr:]
tr_y_1 = train_y.tondarray()[:n_tr]
tr_y_2 = train_y.tondarray()[n_tr:]
tr1 = CDataset(tr_X_1,tr_y_1)
tr2 = CDataset(tr_X_2,tr_y_2)

from collections import namedtuple
CLF = namedtuple('CLF', 'clf_name clf xval_parameters')

from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
# Binary classifiers
from secml.ml.classifiers import CClassifierSVM, CClassifierSGD
# Natively-multiclass classifiers
from secml.ml.classifiers import CClassifierKNN, CClassifierDecisionTree, CClassifierRandomForest

# Let's create a 3-Fold data splitter
from secml.data.splitter import CDataSplitterKFold
xval_splitter = CDataSplitterKFold(num_folds=3, random_state=random_state)

# Metric to use for training and performance evaluation
from secml.ml.peval.metrics import CMetricAccuracy
metric = CMetricAccuracy()
#
# surr_clf = CLF(
#     clf_name='SVM Linear',
#     clf=CClassifierMulticlassOVA(CClassifierSVM, kernel='linear'),
#     xval_parameters={'C': [1e-2, 0.1,  1]})

# surr_clf = CLF(clf_name='Logistic (SGD)',
#         clf=CClassifierMulticlassOVA(
#             CClassifierSGD, regularizer='l2', loss='log',
#             random_state=random_state),
#         xval_parameters={'alpha': [1e-6, 1e-5, 1e-4]})

surr_clf = CLF(clf_name='SVM RBF',
        clf=CClassifierMulticlassOVA(CClassifierSVM, kernel='rbf'),
        xval_parameters={'C': [1e-2, 0.1, 1, 10, 100], 'kernel.gamma': [1, 10, 100, 1000]})


print("Estimating the best training parameters of the surrogate classifier...")
best_params = surr_clf.clf.estimate_parameters(
    dataset=tr1,
    parameters=surr_clf.xval_parameters,
    splitter=xval_splitter,
    metric=metric,
    perf_evaluator='xval'
)

print("The best training parameters of the surrogate classifier are: ",
      [(k, best_params[k]) for k in sorted(best_params)])

surr_clf.clf.fit(tr1.X, tr1.Y)

y_pred = surr_clf.clf.predict(ts.X)

acc = metric.performance_score(y_true=ts.Y, y_pred=y_pred)

print("Accuracy of the surrogate classifier on test set: {:.2%}".format(acc))

print("\nTraining the target classifiers...")

target_clf_list = [
    CLF(
        clf_name='SVM Linear',
        clf=CClassifierMulticlassOVA(CClassifierSVM, kernel='linear'),
        xval_parameters={'C': [1e-2, 0.1, 1]}),
    CLF(clf_name='SVM RBF',
        clf=CClassifierMulticlassOVA(CClassifierSVM, kernel='rbf'),
        xval_parameters={'C': [1e-2, 0.1, 1], 'kernel.gamma': [1, 10, 100]}),
    CLF(clf_name='Logistic (SGD)',
        clf=CClassifierMulticlassOVA(
            CClassifierSGD, regularizer='l2', loss='log',
            random_state=random_state),
        xval_parameters={'alpha': [1e-6, 1e-5, 1e-4]}),
    CLF(clf_name='kNN',
        clf=CClassifierKNN(),
        xval_parameters={'n_neighbors': [30, 40, 50]}),
    CLF(clf_name='Decision Tree',
        clf=CClassifierDecisionTree(random_state=random_state),
        xval_parameters={'max_depth': [1, 3, 5]}),
    CLF(clf_name='Random Forest',
        clf=CClassifierRandomForest(random_state=random_state),
        xval_parameters={'n_estimators': [20, 30, 40]}),
]

for i, test_case in enumerate(target_clf_list):
    clf = test_case.clf
    xval_params = test_case.xval_parameters

    print("\nEstimating the best training parameters of {:} ..."
          "".format(test_case.clf_name))

    best_params = clf.estimate_parameters(
        dataset=tr2, parameters=xval_params, splitter=xval_splitter,
        metric='accuracy', perf_evaluator='xval')

    print("The best parameters for '{:}' are: ".format(test_case.clf_name),
          [(k, best_params[k]) for k in sorted(best_params)])

    print("Training of {:} ...".format(test_case.clf_name))
    clf.fit(tr2.X, tr2.Y)

    # Predictions on test set and performance evaluation
    y_pred = clf.predict(ts.X)
    acc = metric.performance_score(y_true=ts.Y, y_pred=y_pred)

    print("Classifier: {:}\tAccuracy: {:.2%}".format(test_case.clf_name, acc))

noise_type = 'l2'  # Type of perturbation 'l1' or 'l2'
dmax = 0.4  # Maximum perturbation
lb, ub = 0, 1  # Bounds of the attack space. Can be set to `None` for unbounded
y_target = None  # `error-specific` attack. None for `error-generic`

# Should be chosen depending on the optimization problem
solver_params = {
    'eta': 1e-1,
    'eta_min': 0.1,
    'eta_max': None,
    'max_iter': 100,
    'eps': 1e-4
}

from secml.adv.attacks.evasion import CAttackEvasionPGDLS
pgd_ls_attack = CAttackEvasionPGDLS(
    classifier=surr_clf.clf,
    double_init_ds=tr1,
    double_init=False,
    distance=noise_type,
    dmax=dmax,
    lb=lb, ub=ub,
    solver_params=solver_params,
    y_target=y_target)

# Run the evasion attack on x0
print("Attack started...")
print(ts.X)
print(ts.Y)
y_pred, scores, adv_ds, f_obj = pgd_ls_attack.run(ts.X, ts.Y)

print(adv_ds)
print(y_pred)
print("Attack complete!")
# Metric to use for testing transferability
from secml.ml.peval.metrics import CMetricTestError

metric = CMetricTestError()

trans_error = []
transfer_rate = 0.0
for target_clf in target_clf_list:
    print("\nTesting transferability of {:}".format(target_clf.clf_name))

    origin_error = metric.performance_score(
        y_true=ts.Y, y_pred=target_clf.clf.predict(ts.X))

    print("Test error (no attack): {:.2%}".format(origin_error))

    trans_error_clf = metric.performance_score(
        y_true=ts.Y, y_pred=target_clf.clf.predict(adv_ds.X))

    trans_error.append(trans_error_clf)
    transfer_rate += trans_error_clf

# Computing the transfer rate
transfer_rate /= len(target_clf_list)

from secml.array import CArray

trans_acc = CArray(trans_error) * 100  # Show results in percentage

from secml.figure import CFigure
# Only required for visualization in notebooks


fig = CFigure(height=10,width=12)
a = fig.sp.imshow(trans_acc.reshape((1, 6)),
                  cmap='Oranges', interpolation='nearest',
                  alpha=.65, vmin=60, vmax=70)

fig.sp.xticks(CArray.arange((len(target_clf_list))))
fig.sp.xticklabels([c.clf_name for c in target_clf_list],
                   rotation=45, ha="right", rotation_mode="anchor")
fig.sp.yticks([0])
fig.sp.yticklabels([surr_clf.clf_name])

for i in range(len(target_clf_list)):
    fig.sp.text(i, 0, trans_acc[i].round(2).item(), va='center', ha='center')

fig.sp.title("Test error of target classifiers under attack (%)")

plt.savefig("transfer_pois_ridge_source_svmRBF.png")

print("\nAverage transfer rate: {:.2%}".format(transfer_rate))

#
# from secml.figure import CFigure
# from secml.array import CArray
# from math import ceil
#
# fig = CFigure(width=4.5 * len(target_clf_list) / 2,
#               height=4 * 2, markersize=10)
#
# for clf_idx in range(len(target_clf_list)):
#     clf = target_clf_list[clf_idx].clf
#
#     fig.subplot(2, int(ceil(len(target_clf_list) / 2)), clf_idx + 1)
#     fig.sp.title(target_clf_list[clf_idx].clf_name)
#
#     fig.sp.plot_decision_regions(clf, n_grid_points=200)
#     fig.sp.grid(grid_on=False)
#
#     s_idx = ts.Y.find(ts.Y != y_target)
#
#     for pt in s_idx[:10]:  # Plot the translation of multiple adversarial samples
#         pt_segment = CArray.append(ts.X[pt, :], adv_ds.X[pt, :], axis=0)
#         fig.sp.plot_path(pt_segment)
#
#     acc = metric.performance_score(
#         y_true=ts[s_idx[:10], :].Y, y_pred=clf.predict(adv_ds[s_idx[:10], :].X))
#
#     fig.sp.text(0.01, 0.01, "Transfer attack success: {:.1%}".format(acc),
#                 bbox=dict(facecolor='white'))
#
# fig.show()





