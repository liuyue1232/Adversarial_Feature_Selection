from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from lassonet import LassoNetClassifier
from ReliefF import ReliefF
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import datasets
from sklearn import svm
# from mRMRquan import mRMR
from sklearn.metrics import f1_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import secml
from secml.array import CArray
from secml.data import CDataset
from secml.ml.features import CNormalizerMinMax
import matplotlib.pyplot as plt
import pickle
import stability as st
from skfeature.function.similarity_based import fisher_score
from feature_selection_stability_after_adv_training import generateAtificialDataset,getMutualInfos,getBootstrapSample
from mrmr import mrmr_classif
from sklearn.feature_selection import RFE
from sklearn.svm import SVR,SVC

acc_all_methods = []

#归一化
def featureNormaliza(X):
    #X_norm = np.array(X)  # 将X转化为numpy数组对象，才可以进行矩阵的运算
    X_norm = X
    # 定义所需变量
    mu = np.zeros((1, X.shape[1]))
    sigma = np.zeros((1, X.shape[1]))

    mu = np.mean(X_norm, 0)  # 求每一列的平均值（0指定为列，1代表行）
    sigma = np.std(X_norm, 0)  # 求每一列的标准差
    for i in range(X.shape[1]):  # 遍历列
        X_norm[:, i] = (X_norm[:, i] - mu[i]) / sigma[i]  # 归一化

    return X_norm, mu, sigma


# X = pd.read_csv('wdbc.data', header=None)
# X2 = np.array(X.loc[:, :])

from sklearn.datasets import load_breast_cancer
X = load_breast_cancer().data
y = load_breast_cancer().target

# column_headers = list(X.columns.values)
# # print("column:", column_headers)
# X = np.delete(X2, -1, axis=1)
# y = X2[:, -1]
X = np.array(X, dtype=np.float64)
X,mu,sigma3 = featureNormaliza(X)    # x出生就已经归一化了
k = X.shape[1]#特征总数
print(X.shape)
print(y.shape)
print(k)


# 没有特征选择
X_train, X_test, y_train, y_test = train_test_split(X, y)
clf_svc = svm.SVC(kernel='linear', C=1).fit(X_train, y_train.astype('int'))
y_predict_all_features = clf_svc.predict(X_test)
print("特征选择之前 f1:", f1_score(y_predict_all_features, y_test.astype('int'), average='macro'))  # F1值一般越大越好
acc_all_features = clf_svc.score(X_test, y_test.astype('int'))
print("特征选择之前 acc :",acc_all_features)

iter_n = X_train.shape[0]
y_train = y_train.astype(int)
y_test = y_test.astype(int)
# 特征选择之前的攻击
# 数据格式统一
train_y = CArray(y_train)
train_x = CArray(X_train)
tr_X = train_x
test_x = CArray(X_test)
test_y = CArray(y_test)

# 训练与评价指标
from secml.ml.peval.metrics import CMetricAccuracy
metric = CMetricAccuracy()

# 分类器的创建
from secml.ml.classifiers import CClassifierRidge
from secml.ml.classifiers import CClassifierSVM
from secml.ml.classifiers import CClassifierLogistic
from secml.ml.classifiers import CClassifierRandomForest
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.ml.kernels import CKernelRBF
from secml.ml.kernels import CKernelLinear
# clf = CClassifierSVM(kernel=CKernelLinear, C=1)
clf = CClassifierMulticlassOVA(CClassifierSVM, kernel=CKernelLinear())

# clf = CClassifierRidge()
# clf = CClassifierLogistic()
# clf = CClassifierRandomForest()
# We can now fit the classifier
clf.fit(tr_X, train_y)
print("Training of classifier complete!")

# 测试集
y_pred = clf.predict(test_x)
acc = metric.performance_score(y_true=test_y, y_pred=y_pred)
print("Accuracy on test set: {:.2%}".format(acc))

# CDataset格式
tr = CDataset(train_x, train_y)
ts = CDataset(test_x, test_y)

# 正常样本下的stability 评价-----------------------------------------------------------------------------------------------
import random

m = 10
d = 30
Z = np.zeros((m, d))

print(y_train.shape)
print(X_train.shape)
y_train_ = y_train.reshape(-1,1)
train = np.concatenate((X_train, y_train_),axis = 1)

# 对抗训练下的stability 评价-----------------------------------------------------------------------------------------------
# 分类器的创建
from secml.ml.classifiers import CClassifierRidge
from secml.ml.classifiers import CClassifierSVM
from secml.ml.classifiers import CClassifierLogistic
from secml.ml.classifiers import CClassifierRandomForest
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.ml.kernels import CKernelRBF
from secml.ml.kernels import CKernelLinear
from secml.adv.attacks.evasion import CAttackEvasionPGDLS
# clf = CClassifierSVM(kernel=CKernelLinear, C=1)
clf = CClassifierMulticlassOVA(CClassifierSVM, kernel=CKernelLinear())

# clf = CClassifierRidge()
# clf = CClassifierLogistic()
# clf = CClassifierRandomForest()
# We can now fit the classifier
clf.fit(tr_X, train_y)
print("Training of classifier complete!")


noise_type = 'l2'  # Type of perturbation 'l1' or 'l2'
# 攻击强度
dmax = [0, 0.5, 1, 1.5, 2, 2.5]
acc_selected_features_pgd_rf = []
lb, ub = 0, 1  # Bounds of the attack space. Can be set to `None` for unbounded
y_target = None  # None if `error-generic` or a class label for `error-specific`

# Should be chosen depending on the optimization problem
solver_params = {
    'eta': 0.3,
    'eta_min': 0.1,
    'eta_max': None,
    'max_iter': 100,
    'eps': 1e-4
}

stab_random_forest = []
stab_random_forest_adv = []

acc_fs_rf = []
acc_fs_rf_adv = []

Z_before_adv = np.zeros((m, d))
Z_after_adv = np.zeros((m, d))

print(iter_n)

for t in range(len(dmax)):
    pgd_ls_attack = CAttackEvasionPGDLS(classifier=clf, double_init_ds=tr, double_init=False,
                                        distance=noise_type, dmax=dmax[t], lb=lb, ub=ub, solver_params=solver_params,
                                        y_target=y_target)

    # Run the evasion attack on x0
    y_pred_pgdls, _, adv_ds_pgdls, _ = pgd_ls_attack.run(ts.X, ts.Y)
    print("dmax = ",t)
    print("----------------------------------------------------------------------------------------------------------")
    for i in range(m):
        subsample_x_array = np.zeros([iter_n, 30], dtype=float)
        subsample_y_array = []
        # rf = RandomForestRegressor()
        # rf_adv = RandomForestRegressor()
        # rf = Ridge(alpha=0.5)
        # rf_adv = Ridge(alpha=0.5)
        # rf = linear_model.Lasso(alpha=0.003)
        # rf_adv = linear_model.Lasso(alpha=0.003)
        # train_subsample = random.sample(list(train), 20)

        # X = np.array(train_subsample)[:, :-1]
        # y = np.array(train_subsample)[:, -1]
        for m_i in range(iter_n):
            rd = random.randint(0, X_train.shape[0] - 1)
            subsample_x_array[m_i] = X_train[rd]
            subsample_y_array.append(y_train[rd])
        X = subsample_x_array
        y = subsample_y_array
        y = np.array(y)
        # print(X)
        # print(y)
        X_subsample = CArray(X)
        y_subsample = CArray(y.astype(int))
        train_subsample_ = CDataset(X_subsample, y_subsample)

        # RFE
        # estimator = SVC(kernel="linear", C=1)
        # selector = RFE(estimator, n_features_to_select=15, step=1)
        # selector = selector.fit(X, y)
        # w_randomForest = selector.ranking_
        # w_randomForest = fisher_score.fisher_score(X, y) #fisher_score
        #
        # # 正常样本特征选择
        # rf.fit(X, y)
        # w_randomForest = rf.feature_importances_ #rf
        # w_randomForest = rf.coef_ #Lasso
        # key = [i for i in range(X.shape[1])]
        # score_inial = dict(map(lambda x, y: [x, y], key, w_randomForest))
        # print("score_inial:", score_inial)
        # score_sorted = sorted(score_inial.items(), key=lambda x: x[1], reverse=True)  # RF选出的特征重要性分数 , RFE的reverse=False
        # print("dic3.keys:", score_sorted)
        # 选择20个特征
        # top_k = 20
        # RFselec = [i[0] for i in score_sorted][: top_k]
        # RFselec.sort()
        # mrMR
        selected_features = mrmr_classif(X, y, K = 20)
        RFselec = selected_features

        # LassoNet
        # n_selected = []
        # accuracy = []
        # selected = []
        # RFselec = []
        # model = LassoNetClassifier(eps_start=1e-2, lambda_start=1000)
        # path = model.path(X, y)
        # for save in path:
        #     model.load(save.state_dict)
        #     n_selected.append(save.selected.sum())
        #     selected.append(save.selected.numpy().tolist())
        # for i in range (len(selected[1])):
        #     if selected[1][i] == 1:
        #         RFselec.append(i)
        # if len(RFselec) == 0:
        #     for i in range(len(selected[0])):
        #         if  selected[0][i] == 1:
        #             RFselec.append(i)
        print("RFselec:", RFselec)
        # 没有逃避攻击
        X_train_feature_selection_rf = X[:, RFselec]
        # X_test_feature_selection_rf = X_test[:, RFselec]
        clf_fs_rf = svm.SVC(kernel='linear', C=1).fit(X_train_feature_selection_rf, y.astype('int'))
        # y_predict_feature_selection_rf = clf_fs_rf.predict(X_test_feature_selection_rf)
        # # print("f1_RF特征选择后:", f1_score(y_predict_feature_selection_rf, y_test.astype('int'), average='macro'))  # F1值一般越大越好
        # acc_fs_rf_m = clf_fs_rf.score(X_test_feature_selection_rf, y_test.astype('int'))
        # # print("accRF特征选择后:",acc_fs_rf)
        # acc_fs_rf.append(acc_fs_rf_m)
        # x0, y0 = ts[10, :].X, ts[10, :].Y  # Initial sample

        # pgd_ls_attack = CAttackEvasionPGDLS(classifier=clf,double_init_ds=train_subsample_,double_init=False,distance=noise_type,dmax=dmax[t],lb=lb, ub=ub,solver_params=solver_params,y_target=y_target)
        #
        # # Run the evasion attack on x0
        # y_pred_pgdls, _, adv_ds_pgdls, _ = pgd_ls_attack.run(ts.X, ts.Y)
        test_x_all_features_pgd = adv_ds_pgdls.X
        test_x_selected_features_pgd_rf = test_x_all_features_pgd.tondarray()[:, RFselec]
        y_predict_feature_selection_pgd_rf = clf_fs_rf.predict(test_x_selected_features_pgd_rf)
        y_pred_pgdls = CArray(y_predict_feature_selection_pgd_rf)
        acc_fs_rf_m = metric.performance_score(y_true=test_y, y_pred=y_pred_pgdls)
        # print("Accuracy on test set: {:.2%}".format(acc_fs_rf_m))
        acc_fs_rf.append(acc_fs_rf_m)



        pgd_ls_attack = CAttackEvasionPGDLS(classifier=clf, double_init_ds=train_subsample_, double_init=False, distance=noise_type, dmax = dmax[t], lb=lb, ub=ub, solver_params=solver_params, y_target=y_target)
        # 对抗训练特征选择
        x0, y0 = X[10, :], y[10]  # Initial sample
        y_pred_pgdls, _, adv_ds_pgdls_, _ = pgd_ls_attack.run(X_subsample, y_subsample)
        X_adv = adv_ds_pgdls_.X.tondarray()
        y_adv = adv_ds_pgdls_.Y.tondarray()

        # mrmr
        selected_features = mrmr_classif(X_adv, y, K=20)
        RFselec_adv = selected_features
        print("RFselec_adv:", RFselec_adv)

        # RFE
        # estimator = SVC(kernel="linear", C=1)
        # selector = RFE(estimator, n_features_to_select=15, step=1)
        # selector = selector.fit(X_adv, y)
        # w_randomForest_adv = selector.ranking_

        # LassoNet
        # n_selected = []
        # accuracy = []
        # selected_ = []
        # RFselec_adv = []
        # model_ = LassoNetClassifier(eps_start=1e-2, lambda_start=1000)
        # path_ = model_.path(X_adv, y)
        # for save in path_:
        #     model.load(save.state_dict)
        #     n_selected.append(save.selected.sum())
        #     selected_.append(save.selected.numpy().tolist())
        # for i in range(len(selected_[1])):
        #     if selected_[1][i] == 1:
        #         RFselec_adv.append(i)
        # if len(RFselec_adv) == 0:
        #     for i in range(len(selected_[0])):
        #         if  selected_[0][i] == 1:
        #             RFselec_adv.append(i)
        # print("RFselec_adv:", RFselec_adv)

        # w_randomForest_adv = fisher_score.fisher_score(X_adv, y)  # fisher_score
        # rf_adv.fit(X_adv, y)
        # w_randomForest_adv = rf_adv.feature_importances_
        # w_randomForest_adv = rf_adv.coef_
        # key = [i for i in range(X_adv.shape[1])]
        # score_inial = dict(map(lambda x, y: [x, y], key, w_randomForest_adv))
        # score_sorted = sorted(score_inial.items(), key=lambda x: x[1], reverse=True)  # RF选出的特征重要性分数 , RFE的reverse=False
        # 选择20个特征
        # top_k = 20
        # RFselec_adv = [i[0] for i in score_sorted][: top_k]
        # RFselec_adv.sort()
        # print("RFselec_adv:", RFselec_adv)

        # 没有逃避攻击
        X_train_feature_selection_rf = X_adv[:, RFselec_adv]
        # X_test_feature_selection_rf = X_test[:, RFselec_adv]
        clf_fs_rf = svm.SVC(kernel='linear', C=1).fit(X_train_feature_selection_rf, y.astype('int'))
        # y_predict_feature_selection_rf = clf_fs_rf.predict(X_test_feature_selection_rf)
        # # print("f1_RF特征选择后:", f1_score(y_predict_feature_selection_rf, y_test.astype('int'), average='macro'))  # F1值一般越大越好
        # acc_fs_rf_adv_m = clf_fs_rf.score(X_test_feature_selection_rf, y_test.astype('int'))
        # # print("accRF特征选择后:",acc_fs_rf)
        # acc_fs_rf_adv.append(acc_fs_rf_adv_m)

        # x0, y0 = ts[10, :].X, ts[10, :].Y  # Initial sample
        #
        # pgd_ls_attack = CAttackEvasionPGDLS(classifier=clf, double_init_ds=train_subsample_, double_init=False,
        #                                     distance=noise_type, dmax=dmax[t], lb=lb, ub=ub,
        #                                     solver_params=solver_params, y_target=y_target)
        #
        # # Run the evasion attack on x0
        # y_pred_pgdls, _, adv_ds_pgdls, _ = pgd_ls_attack.run(ts.X, ts.Y)
        # test_x_all_features_pgd = adv_ds_pgdls.X
        test_x_selected_features_pgd_rf = test_x_all_features_pgd.tondarray()[:, RFselec_adv]
        y_predict_feature_selection_pgd_rf = clf_fs_rf.predict(test_x_selected_features_pgd_rf)
        y_pred_pgdls = CArray(y_predict_feature_selection_pgd_rf)
        acc_fs_rf_adv_m = metric.performance_score(y_true=test_y, y_pred=y_pred_pgdls)
        # print("Accuracy on test set: {:.2%}".format(acc_fs_rf_m))
        acc_fs_rf_adv.append(acc_fs_rf_adv_m)

        for j_1 in range(d):
            if j_1 in RFselec:
                Z_before_adv[i, j_1] = 1
        # print(Z_before_adv)

        for j_2 in range(d):
            if j_2 in RFselec_adv:
                Z_after_adv[i, j_2] = 1
        # print(Z_after_adv)

    # print(Z_before_adv)

    stab_random_forest_m = st.getStability(Z_before_adv)
    stab_random_forest.append(stab_random_forest_m)

    # print(Z_after_adv)
    stab_random_forest_adv_m = st.getStability(Z_after_adv)
    stab_random_forest_adv.append(stab_random_forest_adv_m)

print(stab_random_forest)
print(stab_random_forest_adv)
acc_fs_rf_avg = []
acc_fs_rf_adv_avg = []
for i in range(len(dmax)):
    sum = 0
    for j in range(m):
        sum += acc_fs_rf[i * m + j]
    acc_fs_rf_avg.append(sum/m)

for i in range(len(dmax)):
    sum = 0
    for j in range(m):
        sum += acc_fs_rf_adv[i * m + j]
    acc_fs_rf_adv_avg.append(sum/m)

print(acc_fs_rf_avg)
print(acc_fs_rf_adv_avg)


# 对抗训练下的stability 评价-----------------------------------------------------------------------------------------------

