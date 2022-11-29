import secml

import pandas as pd
import numpy as np
from secml.array import CArray
from secml.data import CDataset
from sklearn.model_selection import train_test_split
from secml.ml.features import CNormalizerMinMax

#导入数据
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




#训练集与验证集的划分
train_x, valid_x, train_y, valid_y = train_test_split(data_array, target_tr_val, test_size=0.25, shuffle=True)
train_y = CArray(train_y)
train_x = CArray(train_x)
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
# clf = CClassifierSVM(kernel=CKernelRBF(gamma=10), C=1)
clf = CClassifierRidge()
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



#生成对抗样本


lb, ub = val.X.min(), val.X.max()  # Bounds of the attack space. Can be set to `None` for unbounded


# Should be chosen depending on the optimization problem
solver_params = {
    'eta': 0.05,
    'eta_min': 0.05,
    'eta_max': None,
    'max_iter': 100,
    'eps': 1e-6
}



from secml.adv.attacks import CAttackPoisoningSVM
from secml.adv.attacks import CAttackPoisoningRidge
from secml.adv.attacks import CAttackPoisoningLogisticRegression

pois_attack = CAttackPoisoningRidge(classifier=clf,
                                  training_data=tr,
                                  val=val,
                                  lb=lb, ub=ub,
                                  solver_params=solver_params,
                                  random_seed=random_state)

# print(pois_attack)

# chose and set the initial poisoning sample features and label
xc = tr[0,:].X
yc = tr[0,:].Y
pois_attack.x0 = xc
pois_attack.xc = xc
pois_attack.yc = yc



print("Initial poisoning sample features: {:}".format(xc.ravel()))
print("Initial poisoning sample label: {:}".format(yc.item()))



from secml.figure import CFigure
# Only required for visualization in notebooks


fig = CFigure(4,5)

grid_limits = [(lb - 0.1, ub + 0.1),
                    (lb - 0.1, ub + 0.1)]

fig.sp.plot_ds(tr)

# highlight the initial poisoning sample showing it as a star
fig.sp.plot_ds(tr[0,:], markers='*', markersize=16)

fig.sp.title('Attacker objective and gradients')



n_poisoning_points = 30  # Number of poisoning points to generate
pois_attack.n_points = n_poisoning_points

# Run the poisoning attack
print("Attack started...")
pois_attack.init_type = 'random'
pois_y_pred, pois_scores, pois_ds, f_opt = pois_attack.run(ts.X, ts.Y)
print(ts.X.tondarray().shape)
print(ts.Y.tondarray().shape)
print(pois_y_pred)
print(pois_ds.X.tondarray().shape)
print(pois_ds.Y.tondarray().shape)
print("Attack complete!")


y_true_label = pois_ds.Y.tondarray()
print(y_true_label)
y_pois_pred  = pois_y_pred.tondarray()

# Evaluate the accuracy of the original classifier
acc = metric.performance_score(y_true=ts.Y, y_pred=y_pred)
# Evaluate the accuracy after the poisoning attack
pois_acc = metric.performance_score(y_true=ts.Y, y_pred=pois_y_pred)



print("Original accuracy on test set: {:.2%}".format(acc))
print("Accuracy after attack on test set: {:.2%}".format(pois_acc))

