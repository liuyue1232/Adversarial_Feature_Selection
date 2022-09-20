import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
# 攻击强度固定：0.5/1.0/1.5/2.0/2.5

# 首先，先做攻击强度为1.5 数据集:Sonar

number_of_selected_features  = ['10', '15', '20', '25', '30']


# random forest
acc_fs_stability_1     = [ 0.3308186819383703,  0.38532804774843,  0.34901162813361253,  0.3178711484593839,   0.25906945956910754]
acc_fs_stability_adv_1 = [ 0.3173613928122171,  0.347887303069376,  0.3481095176010427,  0.30960728838109697,   0.22645204825622522]
acc_fs_rf_avg_1        = [ 0.7115384615384616,  0.7769230769230768,  0.7153846153846154,  0.7211538461538461,   0.6942307692307693]
acc_fs_rf_adv_avg_1    = [ 0.7192307692307692,  0.7923076923076923,  0.6961538461538461,  0.7423076923076923,   0.7000000000000001]

# Lasso
acc_fs_stability_2     = [ 0.505091927196649,  0.5399698340874811,  0.509621702806721,  0.3926190004621376,   0.37629937629937626 ]
acc_fs_stability_adv_2 = [ 0.51164751259102,  0.5291467192340185,  0.4818788826279462,  0.3761140819964348,   0.39527131668608695 ]
acc_fs_rf_avg_2        = [ 0.6480769230769232,  0.7673076923076922,  0.7596153846153846,  0.7096153846153846,   0.6923076923076924]
acc_fs_rf_adv_avg_2    = [ 0.6673076923076924,  0.7653846153846153,  0.7653846153846153,  0.7192307692307693,   0.7096153846153846]

# Ridge
acc_fs_stability_3     = [ 0.44814043640918866,  0.4477210991550733,  0.47813130575740215,   0.39227735175122325, 0.35480990333311035]
acc_fs_stability_adv_3 = [ 0.44306197416755544,  0.4074510203542462,  0.44332210998877675,   0.34225322826359106, 0.3445444915254233]
acc_fs_rf_avg_3        = [ 0.725,  0.6942307692307693,  0.7384615384615385,   0.6942307692307692, 0.75]
acc_fs_rf_adv_avg_3    = [ 0.6961538461538462,  0.6692307692307693,  0.7307692307692307,   0.6846153846153845, 0.7519230769230768]

# RFE
acc_fs_stability_4     = [ 0.12499999999999989,  0.19559377485630247, 0.13575955749516144 ,  0.24011918462581372,   0.30231965981717723 ]
acc_fs_stability_adv_4 = [ 0.09307011562007583,  0.1817249417249418,  0.12717722936701015,  0.25062929272227574,   0.24007444168734493 ]
acc_fs_rf_avg_4        = [ 0.6903846153846154,  0.7096153846153845,  0.7269230769230769,  0.7557692307692307,   0.7134615384615385 ]
acc_fs_rf_adv_avg_4    = [ 0.6749999999999999,  0.7423076923076923,  0.6980769230769232,  0.7730769230769231,   0.7557692307692307 ]

# Fisher_score
acc_fs_stability_5     = [ 0.5978265520044496,  0.6996996996996997,  0.6958310967972803,  0.5756629280835424,   0.6789233823844589 ]
acc_fs_stability_adv_5 = [ 0.5971514214254348,  0.6951571373424548,  0.6976331977895943,  0.5472672658485983,   0.6695156695156694 ]
acc_fs_rf_avg_5        = [ 0.6692307692307693,  0.6980769230769232,  0.7461538461538462,  0.7615384615384615,   0.6596153846153845 ]
acc_fs_rf_adv_avg_5    = [ 0.6903846153846154,  0.7115384615384617,  0.7346153846153846,  0.7538461538461538,   0.6750000000000002 ]

# mrMR
acc_fs_stability_6     = [ 0.5081515976048336,  0.6141656331817555,  0.6110722183329443,  0.6233351072060749,   0.5343101245252946 ]
acc_fs_stability_adv_6 = [ 0.5122212692967409,  0.6262863263175333,  0.6007363044774572,  0.5885194657573173,   0.5114638447971782 ]
acc_fs_rf_avg_6        = [ 0.7192307692307692,  0.7173076923076923,  0.7057692307692307,  0.7326923076923078,   0.7596153846153846 ]
acc_fs_rf_adv_avg_6    = [ 0.7173076923076924,  0.698076923076923,  0.7115384615384616,  0.7326923076923076,   0.7653846153846153 ]

x_major_locator=MultipleLocator(5)
y_major_locator=MultipleLocator(0.1)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
plt.xlim(10, 30)
plt.ylim(0.6, 0.9)  # 坐标刻度


name_list_avg = [acc_fs_rf_avg_2, acc_fs_rf_avg_3, acc_fs_rf_avg_4, acc_fs_rf_avg_6]
name_list_avg_adv = [acc_fs_rf_adv_avg_2, acc_fs_rf_adv_avg_3, acc_fs_rf_adv_avg_4, acc_fs_rf_adv_avg_6]


# 绘制柱状图
total_width, n = 1, 2
width = total_width / n
x = list(range(len(number_of_selected_features)))
plt.bar(x, acc_fs_rf_avg_2, label = 'Lasso', width=width,  fc = 'y')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, acc_fs_rf_adv_avg_2, width=width, label = 'Lasso PGD-AT',tick_label = number_of_selected_features, fc = 'r')
plt.legend()
plt.title("Accuracy")  # 设置标题及字体
plt.ylabel("acc Measure")  # 设置标题及字体
plt.xlabel('number of selected features')
plt.show()




# ln1, = plt.plot(number_of_selected_features, acc_fs_rf_avg_1, color='green', marker="<",label = "random Forest")
# ln2, = plt.plot(number_of_selected_features, acc_fs_rf_adv_avg_1, color='green', linestyle='--', marker=">", label = "random ForestPGD_AT")
# ln3, = plt.plot(number_of_selected_features, acc_fs_rf_avg_2, color='b', marker="<",label = "Lasso")
# ln4, = plt.plot(number_of_selected_features, acc_fs_rf_adv_avg_2, color='b', linestyle='--',marker=">", label = "Lasso PGD_AT")
# ln5, = plt.plot(number_of_selected_features, acc_fs_rf_avg_3, color='r', marker="<",label = "Ridge")
# ln6, = plt.plot(number_of_selected_features, acc_fs_rf_adv_avg_3, color='r', linestyle='--', marker=">", label = "Ridge PGD_AT")
# ln7, = plt.plot(number_of_selected_features, acc_fs_rf_avg_4, color='c', marker="<",label = "RFE")
# ln8, = plt.plot(number_of_selected_features, acc_fs_rf_adv_avg_4, color='c', linestyle='--', marker=">", label = "RFE PGD_AT")
# ln9, = plt.plot(number_of_selected_features, acc_fs_rf_avg_5, color='m', marker="<",label = "Fisher score")
# ln10, = plt.plot(number_of_selected_features, acc_fs_rf_adv_avg_5, color='m', linestyle='--', marker=">", label = "Fisher score PGD_AT")
# ln11, = plt.plot(number_of_selected_features, acc_fs_rf_avg_6, color='y', marker="<",label = "mrMR")
# ln12, = plt.plot(number_of_selected_features, acc_fs_rf_adv_avg_6, color='y', linestyle='--', marker=">", label = "mrMR PGD_AT")
# plt.legend()


#
# x_major_locator=MultipleLocator(5)
# y_major_locator=MultipleLocator(0.2)
# ax=plt.gca()
# # plt.xlim(10, 30)
# # plt.ylim(0, 1)  # 坐标刻度
# ax.xaxis.set_major_locator(x_major_locator)
# ax.yaxis.set_major_locator(y_major_locator)
# plt.xlim(10,30)
# plt.ylim(0,1)
#
# # ln13, = plt.plot(number_of_selected_features, acc_fs_stability_1, color='pink', marker="^", label = "random Forest stability")
# # ln14, = plt.plot(number_of_selected_features, acc_fs_stability_adv_1, color='pink', linestyle='--', marker="v", label = "random Forest + PGD_AT stability")
# # ln15, = plt.plot(number_of_selected_features, acc_fs_stability_2, color='b', marker="^", label = "Lasso stability")
# # ln16, = plt.plot(number_of_selected_features, acc_fs_stability_adv_2, color='blue', linestyle='--', marker="v", label = "Lassp + PGD_AT stability")
# # ln17, = plt.plot(number_of_selected_features, acc_fs_stability_3, color='m', marker="^", label = "Ridge stability")
# # ln18, = plt.plot(number_of_selected_features, acc_fs_stability_adv_3, color='m', linestyle='--', marker="v", label = "Ridge + PGD_AT stability")
# # ln19, = plt.plot(number_of_selected_features, acc_fs_stability_4, color='y', marker="^", label = "RFE stability")
# # ln20, = plt.plot(number_of_selected_features, acc_fs_stability_adv_4, color='y', linestyle='--', marker="v", label = "RFE + PGD_AT stability")
# # ln21, = plt.plot(number_of_selected_features, acc_fs_stability_5, color='c', marker="^", label = "Fisher score stability")
# # ln22, = plt.plot(number_of_selected_features, acc_fs_stability_adv_5, color='c', linestyle='--', marker="v", label = "Fisher score + PGD_AT stability ")
# # ln23, = plt.plot(number_of_selected_features, acc_fs_stability_6, color='green', marker="^", label = "mrMR stability")
# # ln24, = plt.plot(number_of_selected_features, acc_fs_stability_adv_6, color='green', linestyle='--', marker="v", label = "mrMR + PGD_AT stability")
# # # plt.legend()
#
# plt.title("Feature selection Stability")  # 设置标题及字体
# plt.ylabel("stability Measure")  # 设置标题及字体
# plt.xlabel('number of selected features')
# plt.show()