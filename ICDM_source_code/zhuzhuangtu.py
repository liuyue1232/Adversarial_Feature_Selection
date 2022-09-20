import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings = False
# 使文字可以展示
plt.rcParams['font.sans-serif'] = ['SimHei']
# 使负号可以展示
plt.rcParams['axes.unicode_minus'] = False


# 定义函数来显示柱状图
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()


name = ['10', '15', '20', '25', '30']  # 横坐标
# 纵坐标

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




if __name__ == '__main__':
    plt.figure(figsize=(10, 10))
    total_width, n = 0.2, 2
    width = total_width / n
    x = [0, 1, 2, 3, 4]

    # a_1 = plt.bar(x, acc_fs_rf_avg_2, width=width, label='Lasso', fc='violet', hatch = '/')
    # for i in range(len(x)):
    #     x[i] = x[i] + width
    #
    # b_1 = plt.bar(x, acc_fs_rf_adv_avg_2, width=width, label='Lasso PGD-AT', tick_label=name, fc='blue', hatch = '/')
    # for i in range(len(x)):
    #     x[i] = x[i] + width
    #
    # a_2 = plt.bar(x, acc_fs_rf_avg_3, width=width, label='Ridge', fc='tomato', hatch = '|')
    # for i in range(len(x)):
    #     x[i] = x[i] + width
    #
    # b_2 = plt.bar(x, acc_fs_rf_adv_avg_3, width=width, label='Ridge PGD-AT', tick_label=name, fc='r', hatch = '|')
    # for i in range(len(x)):
    #     x[i] = x[i] + width
    #
    # a_3 = plt.bar(x, acc_fs_rf_avg_4, width=width, label='RFE', fc='pink', hatch = '-')
    # for i in range(len(x)):
    #     x[i] = x[i] + width
    #
    # b_3 = plt.bar(x, acc_fs_rf_adv_avg_4, width=width, label='RFE PGD-AT', tick_label=name, fc='hotpink', hatch = '-')
    # for i in range(len(x)):
    #     x[i] = x[i] + width
    #
    # a_4 = plt.bar(x, acc_fs_rf_avg_6, width=width, label='mrMR', fc='lightskyblue', hatch = '+')
    # for i in range(len(x)):
    #     x[i] = x[i] + width
    #
    # b_4 = plt.bar(x, acc_fs_rf_adv_avg_6, width=width, label='mrMR PGD-AT', tick_label=name, fc='steelblue', hatch = '+')
    # for i in range(len(x)):
    #     x[i] = x[i] + width

    a_1 = plt.bar(x, acc_fs_stability_2, width=width, label='Lasso', fc='violet',hatch = '/')
    for i in range(len(x)):
        x[i] = x[i] + width

    b_1 = plt.bar(x, acc_fs_stability_adv_2, width=width, label='Lasso PGD-AT', tick_label=name, fc='blue',hatch = '/')
    for i in range(len(x)):
        x[i] = x[i] + width

    a_2 = plt.bar(x, acc_fs_stability_3, width=width, label='Ridge', fc='tomato',hatch = '|')
    for i in range(len(x)):
        x[i] = x[i] + width

    b_2 = plt.bar(x, acc_fs_stability_adv_3, width=width, label='Ridge PGD-AT', tick_label=name, fc='r',hatch = '|')
    for i in range(len(x)):
        x[i] = x[i] + width

    a_3 = plt.bar(x, acc_fs_stability_4, width=width, label='RFE', fc='pink',hatch = '-')
    for i in range(len(x)):
        x[i] = x[i] + width

    b_3 = plt.bar(x, acc_fs_stability_adv_4, width=width, label='RFE PGD-AT', tick_label=name, fc='hotpink',hatch = '-')
    for i in range(len(x)):
        x[i] = x[i] + width

    a_4 = plt.bar(x, acc_fs_stability_6, width=width, label='mrMR', fc='lightskyblue',hatch = '+')
    for i in range(len(x)):
        x[i] = x[i] + width

    b_4 = plt.bar(x, acc_fs_stability_adv_6, width=width, label='mrMR PGD-AT', tick_label=name, fc='steelblue',hatch = '+')
    for i in range(len(x)):
        x[i] = x[i] + width

    autolabel(a_1)
    autolabel(b_1)
    autolabel(a_2)
    autolabel(b_2)
    autolabel(a_3)
    autolabel(b_3)
    autolabel(a_4)
    autolabel(b_4)

    # # 鲁棒性
    # plt.title('Robustness of feature selection with or without PGD-AT', fontsize=20)
    # plt.xlabel('Number of selected features', fontsize=20)
    # plt.ylabel('Accuracy Measure', fontsize=20)
    # plt.tick_params(labelsize=20)
    # # plt.legend()
    # plt.ylim(0.2, 1)  # 坐标刻度
    # plt.legend(bbox_to_anchor=(0, 0), loc=3, borderaxespad=0, fontsize=14)
    # plt.show()

    # 稳定性
    # 设置x y 轴标签
    plt.title('Stability of feature selection with or without PGD-AT',fontsize = 20)
    plt.xlabel('Number of selected features', fontsize=20)
    plt.ylabel('Stability Measure', fontsize=20)
    plt.tick_params(labelsize=20)

    # plt.legend()
    plt.ylim(0, 0.7)  # 坐标刻度
    plt.legend(bbox_to_anchor=(0, 0), loc=3, borderaxespad=0,fontsize = 14)
    plt.show()