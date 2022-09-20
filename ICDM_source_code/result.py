import matplotlib.pyplot as plt

attack_length          = [0, 0.5, 1, 1.5, 2, 2.5]
acc_fs_stability_1     = [0.5536486486486487, 0.5523344947735193, 0.5637293664129683, 0.5500648307932071, 0.5576121323589147, 0.5584111439938719]
acc_fs_stability_adv_1 = [0.575945945945946, 0.5248835134331318, 0.5398238832145636, 0.5021776962216418, 0.4732893157262905, 0.45315232968843455]
acc_fs_rf_avg_1        = [0.8851982618142312, 0.8841933731667572, 0.8799293862031503, 0.8699076588810429, 0.8099402498642043, 0.6709397066811515]
acc_fs_rf_adv_avg_1    = [0.8850353068984246, 0.8834057577403585, 0.8821835958718088, 0.8779467680608365, 0.8379413362303098, 0.8196632265073328]




acc_fs_stability_2     =  [0.794054054054054, 0.8182842653894915, 0.8538147566718995, 0.8798268312101911, 0.8876613473198676, 0.8972348634744757]
acc_fs_stability_adv_2 = [0.794054054054054, 0.8294946673031516, 0.8483811230396966, 0.8685983827493261, 0.8733893557422969, 0.8792486583184258]
acc_fs_rf_avg_2        = [0.8173275393807714, 0.8227865290602935, 0.817001629549158, 0.7798479087452472, 0.7392992938620313, 0.6380499728408474]
acc_fs_rf_adv_avg_2    = [0.8173275393807714, 0.8236284627919609, 0.818739815317762, 0.8110537751222159, 0.7922324823465507, 0.7743346007604563]


acc_fs_stability_3     = [0.20540540540540542, 0.2576290989116965, 0.2818342151675486, 0.29079805649247414, 0.29963592283048357, 0.3171591420428982]
acc_fs_stability_adv_3 = [0.20540540540540542, 0.2486357782780697, 0.28496650266406665, 0.291212836243511, 0.28376428037350687, 0.23902188946859648]
acc_fs_rf_avg_3        = [0.839516567083107, 0.8448940793047257, 0.8315317762085824, 0.7960076045627377, 0.70458989679522, 0.648696360673547]
acc_fs_rf_adv_avg_3    = [0.839516567083107, 0.8391906572514938, 0.8357142857142857, 0.8376697447039652, 0.8184953829440522, 0.7619228680065182]

acc_fs_stability_4     = [0.09108108108108093, 0.19024795103226488, 0.2474739717866733, 0.25993825290801165, 0.2737457588642038, 0.2992907801418436]
acc_fs_stability_adv_4 = [0.09108108108108093, 0.19861235898971752, 0.2527408575504996, 0.2743714618714619, 0.21897233201581034, 0.16920314253647561]
acc_fs_rf_avg_4        = [0.7783269961977186, 0.7584464964693101, 0.7639326453014667, 0.7729494839761, 0.7404671374253123, 0.6739272134709396]
acc_fs_rf_adv_avg_4    = [0.7783269961977186, 0.7438348723519826, 0.7691472026072785, 0.7509505703422054, 0.7412275936990765, 0.7135795763172188]



acc_fs_stability_5     = [0.567027027027027, 0.6770909632645967, 0.7007978723404256, 0.7249350721712671, 0.7283860949709478, 0.7204842413121133]
acc_fs_stability_adv_5 = [0.567027027027027, 0.6797799379678415, 0.7000591016548463, 0.715425145180203, 0.7107822410147991, 0.6261579500444588]
acc_fs_rf_avg_5        = [0.8658881042911462, 0.8569527430744162, 0.8487506789788158, 0.8347365562194462, 0.7510863661053775, 0.6671374253123303]
acc_fs_rf_adv_avg_5    = [0.8658881042911462, 0.8571156979902227, 0.8468766974470399, 0.8483161325366648, 0.8295219989136339, 0.8069527430744163]


acc_fs_stability_6     = [0.5131081081081081, 0.6001848067745947, 0.6017556017556018, 0.5877618553520849, 0.5957769259179073, 0.5898903195040535]
acc_fs_stability_adv_6 = [0.5131081081081081, 0.592324431746071, 0.6215969619254291, 0.6035716626457893, 0.5623907813902671, 0.5154309752052744]
acc_fs_rf_avg_6        = [0.8851982618142316, 0.8779467680608365, 0.8791689299293862, 0.86352525801195, 0.8016567083107006, 0.6761542639869635]
acc_fs_rf_adv_avg_6    = [0.8851982618142316, 0.8778381314502989, 0.8799837045084195, 0.8737914177077675, 0.8431558935361216, 0.8427756653992393]


plt.xlim(0, 2.5)
plt.ylim(0, 1)  # 坐标刻度
ln1, = plt.plot(attack_length, acc_fs_rf_avg_1, color='green', marker="<",label = "random Forest")
ln2, = plt.plot(attack_length, acc_fs_rf_adv_avg_1, color='green', linestyle='--', marker=">", label = "random Forest + PGD_AT")
ln3, = plt.plot(attack_length, acc_fs_rf_avg_2, color='b', marker="<",label = "Lasso")
ln4, = plt.plot(attack_length, acc_fs_rf_adv_avg_2, color='b', linestyle='--',marker=">", label = "Lasso + PGD_AT")
ln5, = plt.plot(attack_length, acc_fs_rf_avg_3, color='r', marker="<",label = "Ridge")
ln6, = plt.plot(attack_length, acc_fs_rf_adv_avg_3, color='r', linestyle='--', marker=">", label = "Ridge + PGD_AT")
ln7, = plt.plot(attack_length, acc_fs_rf_avg_4, color='c', marker="<",label = "RFE")
ln8, = plt.plot(attack_length, acc_fs_rf_adv_avg_4, color='c', linestyle='--', marker=">", label = "RFE + PGD_AT")
ln9, = plt.plot(attack_length, acc_fs_rf_avg_5, color='m', marker="<",label = "Fisher score")
ln10, = plt.plot(attack_length, acc_fs_rf_adv_avg_5, color='m', linestyle='--', marker=">", label = "Fisher score + PGD_AT")
ln11, = plt.plot(attack_length, acc_fs_rf_avg_6, color='y', marker="<",label = "mrMR")
ln12, = plt.plot(attack_length, acc_fs_rf_adv_avg_6, color='y', linestyle='--', marker=">", label = "mrMR + PGD_AT")
plt.legend()
plt.title("Accuracy")  # 设置标题及字体
plt.ylabel("acc Measure")  # 设置标题及字体
plt.xlabel('pdg-Perturbation(Attack Strengh)')
plt.show()
# plt.xlim(0, 2.5)
# plt.ylim(0, 1)  # 坐标刻度
# ln13, = plt.plot(attack_length, acc_fs_stability_1, color='pink', marker="^", label = "random Forest stability")
# ln14, = plt.plot(attack_length, acc_fs_stability_adv_1, color='pink', linestyle='--', marker="v", label = "random Forest + PGD_AT stability")
# ln15, = plt.plot(attack_length, acc_fs_stability_2, color='b', marker="^", label = "Lasso stability")
# ln16, = plt.plot(attack_length, acc_fs_stability_adv_2, color='blue', linestyle='--', marker="v", label = "Lassp + PGD_AT stability")
# ln17, = plt.plot(attack_length, acc_fs_stability_3, color='m', marker="^", label = "Ridge stability")
# ln18, = plt.plot(attack_length, acc_fs_stability_adv_3, color='m', linestyle='--', marker="v", label = "Ridge + PGD_AT stability")
# ln19, = plt.plot(attack_length, acc_fs_stability_4, color='y', marker="^", label = "RFE stability")
# ln20, = plt.plot(attack_length, acc_fs_stability_adv_4, color='y', linestyle='--', marker="v", label = "RFE + PGD_AT stability")
# ln21, = plt.plot(attack_length, acc_fs_stability_5, color='c', marker="^", label = "Fisher score stability")
# ln22, = plt.plot(attack_length, acc_fs_stability_adv_5, color='c', linestyle='--', marker="v", label = "Fisher score + PGD_AT stability ")
# ln23, = plt.plot(attack_length, acc_fs_stability_6, color='green', marker="^", label = "mrMR stability")
# ln24, = plt.plot(attack_length, acc_fs_stability_adv_6, color='green', linestyle='--', marker="v", label = "mrMR + PGD_AT stability")
# plt.legend()
# plt.title("Feature selection Stability")  # 设置标题及字体
# plt.ylabel("stability Measure")  # 设置标题及字体
# plt.xlabel('pdg-Perturbation(Attack Strengh)')
# plt.show()


# name_list_rob = [acc_fs_rf_avg_1, acc_fs_rf_avg_2, acc_fs_rf_avg_3, acc_fs_rf_avg_4, acc_fs_rf_avg_5, acc_fs_rf_avg_6]
# name_list_rob_pgd = [acc_fs_rf_adv_avg_1, acc_fs_rf_adv_avg_2, acc_fs_rf_adv_avg_3, acc_fs_rf_adv_avg_4, acc_fs_rf_adv_avg_5, acc_fs_rf_adv_avg_6]
# method_name_list = ["Random Forest", "Lasso", "Ridge", "RFE", "Fisher score", "mrMR"]
# for u, v in enumerate(name_list_rob):
#     for i, j in enumerate(name_list_rob[u]):
#         print(method_name_list[i], "%.2f" % j)
#     print("\n")
#
# for u, v in enumerate(name_list_rob_pgd):
#     for i, j in enumerate(name_list_rob[u]):
#         print(method_name_list[i], "+ PGD-AT", "%.2f" % j)
#     print("\n")
