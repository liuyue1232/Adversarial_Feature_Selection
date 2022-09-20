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


name = ['Spambase', 'Sonar', 'Breast', 'Parkinson', 'Heart', 'Ionosphere']  # 横坐标
# 纵坐标


# Lasso
acc_fs_stability_2     = [0.72, 0.44200766198434327, 0.7421472655837305, 0.4702228615671003, 0.8564132682957984, 1.0]
acc_fs_stability_adv_2 = [0.71, 0.35515873015873, 0.6864175394032184, 0.45472322974446006, 0.8411581617587185, 0.9793419960513021]
acc_fs_stability_adv_combine_2 = [0.71, 0.4678862771828758, 0.7307285903166252, 0.48010105729194363, 0.8432758776250839, 1.0]
acc_fs_rf_avg_2        = [0.78, 0.6423076923076922, 0.8741724941724942, 0.8163265306122449, 0.6578947368421053, 0.8181818181818181]
acc_fs_rf_adv_avg_2    = [0.87, 0.7192307692307691, 0.9093240093240094, 0.8612244897959183, 0.6763157894736842, 0.9204545454545455]
acc_fs_rf_adv_avg_combine_2 = [0.86, 0.6961538461538461, 0.8940750417995325, 0.8275958871118287, 0.664107038845467, 0.8920945470576628]

# Ridge
acc_fs_stability_3     = [0.46, 0.4742271471831222, 0.5555020434968375, 0.6090255399977941, 0.8756379675212291, 0.35805039076217575]
acc_fs_stability_adv_3 = [0.47, 0.41864941864941885, 0.3773797898638236, 0.6017900934967603, 0.8651200658473889, 0.34547306828868847]
acc_fs_stability_adv_combine_3 = [0.49, 0.47254947254947266, 0.5461555451159373, 0.6060927550096489, 0.8717356469179468, 0.3658715695595478]
acc_fs_rf_avg_3        = [0.80, 0.7096153846153845, 0.9044289044289044, 0.8224489795918368, 0.6236842105263158, 0.8954545454545455]
acc_fs_rf_adv_avg_3    = [0.87, 0.7307692307692307, 0.9396270396270396, 0.846938775510204, 0.6657894736842105, 0.9507575757575758]
acc_fs_rf_adv_avg_combine_3 = [0.87, 0.742617535530085, 0.9388294975326303, 0.8385730258155949, 0.6732558268265771, 0.9102403059443462]

# RFE
acc_fs_stability_4     = [0.60, 0.3078311659940445, 0.3778439960459168, 0.47326497768130266, 0.9118629596042875, 0.19480885700748266]
acc_fs_stability_adv_4 = [0.59, 0.22856189522856152, 0.22421956511223043, 0.408395790809016, 0.9027570192483472, 0.17936516188495108]
acc_fs_stability_adv_combine_4 = [0.60, 0.32714904143475576, 0.3539231741408648, 0.47253994552688816, 0.911391783166584, 0.18395488855156633]
acc_fs_rf_avg_4        = [0.86, 0.7057692307692307, 0.9256410256410257, 0.7306122448979592, 0.6236842105263157, 0.8727272727272727]
acc_fs_rf_adv_avg_4    = [0.89, 0.7673076923076922, 0.9307692307692308, 0.8020408163265305, 0.7236842105263158, 0.8818181818181818]
acc_fs_rf_adv_avg_combine_4 = [0.88, 0.7270364599789803, 0.9305237439428062, 0.7731677518045813, 0.6887881906004593, 0.8734128715351838]

# mrMR
acc_fs_stability_6     = [0.6221136341860687, 0.9706939731288802, 0.6774945943703161, 0.8447041782526588, 0.3929838416542166]
acc_fs_stability_adv_6 = [0.4195308237861426, 0.9642739024859749, 0.6687999116461831, 0.8262695775413401, 0.3462815207800893]
acc_fs_stability_adv_combine_6 = [0.6221136341860687, 0.62, 0.94, 0.64, 0.83, 0.37]
acc_fs_rf_avg_6        = [0.6942307692307692, 0.9382284382284383, 0.7224489795918367, 0.5394736842105263, 0.9136363636363635]
acc_fs_rf_adv_avg_6    = [0.7423076923076923, 0.9519813519813519, 0.7673469387755102, 0.6710526315789473, 0.9409090909090908]
acc_fs_rf_adv_avg_combine_6 = [0.7021788118079088, 0.9455383518148753, 0.7666851865949452, 0.6708480377655016, 0.9345619447689096]



if __name__ == '__main__':
    plt.figure(figsize=(10, 10))
    total_width, n = 0.3, 4
    width = total_width / n
    x = [0, 1, 2, 3, 4, 5]

    a_1 = plt.bar(x, acc_fs_rf_avg_2, width=width, label='ℓ1', fc='violet', hatch = '+')
    for i in range(len(x)):
        x[i] = x[i] + width

    b_1 = plt.bar(x, acc_fs_rf_adv_avg_2, width=width,  label='AT-in-processing-form1-ℓ1', fc='blue', hatch = '+')
    for i in range(len(x)):
        x[i] = x[i] + width

    c_1 = plt.bar(x, acc_fs_rf_adv_avg_combine_2, width=width, label='AT-in-processing-form2-ℓ1', fc='lightgreen', hatch = '+')
    for i in range(len(x)):
        x[i] = x[i] + width

    a_2 = plt.bar(x, acc_fs_rf_avg_3, width=width, tick_label=name, label='ℓ2', fc='tomato', hatch = '||')
    for i in range(len(x)):
        x[i] = x[i] + width

    b_2 = plt.bar(x, acc_fs_rf_adv_avg_3, width=width,  label='AT-in-processing-form1-ℓ2', fc='r', hatch = '||')
    for i in range(len(x)):
        x[i] = x[i] + width

    c_2 = plt.bar(x, acc_fs_rf_adv_avg_combine_3, width=width, label='AT-in-processing-form2-ℓ2', fc='turquoise', hatch = '||')
    for i in range(len(x)):
        x[i] = x[i] + width

    a_3 = plt.bar(x, acc_fs_rf_avg_4, width=width, label='RFE', fc='pink', hatch = '-')
    for i in range(len(x)):
        x[i] = x[i] + width

    b_3 = plt.bar(x, acc_fs_rf_adv_avg_4, width=width, label='AT-post-processing-form1-RFE', fc='m', hatch = '-')
    for i in range(len(x)):
        x[i] = x[i] + width

    c_3 = plt.bar(x, acc_fs_rf_adv_avg_combine_4, width=width, label='AT-post-processing-form2-RFE', fc='salmon', hatch = '-')
    for i in range(len(x)):
        x[i] = x[i] + width

    # a_1 = plt.bar(x, acc_fs_stability_2, width=width, label='ℓ1', fc='violet', hatch='+')
    # for i in range(len(x)):
    #     x[i] = x[i] + width
    #
    # b_1 = plt.bar(x, acc_fs_stability_adv_2, width=width, label='AT-in-processing-form1-ℓ1', fc='blue', hatch='+')
    # for i in range(len(x)):
    #     x[i] = x[i] + width
    #
    # c_1 = plt.bar(x, acc_fs_stability_adv_combine_2, width=width, label='AT-in-processing-form2-ℓ1', fc='lightgreen', hatch='+')
    # for i in range(len(x)):
    #     x[i] = x[i] + width
    #
    # a_2 = plt.bar(x, acc_fs_stability_3, width=width, label='ℓ2', tick_label=name,fc='tomato', hatch='||')
    # for i in range(len(x)):
    #     x[i] = x[i] + width
    #
    # b_2 = plt.bar(x, acc_fs_stability_adv_3, width=width, label='AT-in-processing-form1-ℓ2',  fc='r', hatch='||')
    # for i in range(len(x)):
    #     x[i] = x[i] + width
    #
    # c_2 = plt.bar(x, acc_fs_stability_adv_combine_3, width=width, label='AT-in-processing-form2-ℓ2', fc='turquoise', hatch='||')
    # for i in range(len(x)):
    #     x[i] = x[i] + width
    #
    # a_3 = plt.bar(x, acc_fs_stability_4, width=width, label='RFE', fc='pink', hatch='-')
    # for i in range(len(x)):
    #     x[i] = x[i] + width
    #
    # b_3 = plt.bar(x, acc_fs_stability_adv_4, width=width, label='AT-post-processing-form1-RFE', fc='m', hatch='-')
    # for i in range(len(x)):
    #     x[i] = x[i] + width
    #
    # c_3 = plt.bar(x, acc_fs_stability_adv_combine_4, width=width, label='AT-post-processing-form2-RFE', fc='salmon', hatch='-')
    # for i in range(len(x)):
    #     x[i] = x[i] + width

    autolabel(a_1)
    autolabel(b_1)
    autolabel(a_2)
    autolabel(b_2)
    autolabel(a_3)
    autolabel(b_3)


    # 鲁棒性
    plt.title('Robustness of feature selection in different datasets', fontsize=20)
    # plt.xlabel('Number of selected features', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.tick_params(labelsize=20)
    # plt.legend()
    plt.ylim(0.4, 1)  # 坐标刻度
    plt.legend(bbox_to_anchor=(0, 0), loc=3, borderaxespad=0, fontsize=14)
    plt.show()

    # 稳定性
    # 设置x y 轴标签
    # plt.title('Stability of feature selection in different datasets',fontsize = 20)
    # # plt.xlabel('Number of selected features', fontsize=20)
    # plt.ylabel('Stability', fontsize=20)
    # plt.tick_params(labelsize=20)
    #
    # plt.legend()
    # plt.ylim(0.2, 1.0)  # 坐标刻度
    # plt.legend(bbox_to_anchor=(0, 1), loc=2, borderaxespad=0, fontsize=14)
    # plt.show()