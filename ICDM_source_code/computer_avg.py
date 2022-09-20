import matplotlib.pyplot as plt
import warnings
import statistics
# lasso
# ridge
# RFE
# mrMR
acc_fs_stability = []
acc_fs_stability_adv = []
acc_fs_rf_avg = []
acc_fs_rf_adv_avg = []



acc_fs_stability_scikit_breast = [1.0, 1.0, 0.9558477237048666, 0.9558477237048666, 0.9562341956817739, 0.9562341956817739]
acc_fs_stability_adv_scikit_breast = [1.0, 1.0, 0.9558477237048666, 0.9562341956817739, 0.9434069043576684, 0.9301545911715403]
acc_fs_rf_avg_scikit_breast = [0.9622377622377621, 0.9538461538461538, 0.9566433566433566, 0.9454545454545455, 0.9342657342657343, 0.8769230769230768]
acc_fs_rf_adv_avg_scikit_breast = [0.9622377622377621, 0.9594405594405593, 0.9552447552447552, 0.951048951048951, 0.944055944055944, 0.9398601398601398]


acc_fs_stability_Parkinson = [0.5192592592592591, 0.6156276718337608, 0.693678641047062, 0.7025589405019195, 0.7632391163092918, 0.770603937270604]
acc_fs_stability_adv_Parkinson = [0.5192592592592591, 0.5747369986245865, 0.6733387326931235, 0.686979686979687, 0.738895715831154, 0.8195890764892888]
acc_fs_rf_avg_Parkinson = [0.789795918367347, 0.7714285714285714, 0.7755102040816327, 0.7612244897959182, 0.7448979591836734, 0.7224489795918367]
acc_fs_rf_adv_avg_Parkinson = [0.789795918367347, 0.7714285714285715, 0.7816326530612244, 0.7795918367346938, 0.7755102040816326, 0.7673469387755102]


acc_fs_stability_Heart_Disease = [0.78, 0.8, 0.818121693121693, 0.8541114058355438, 0.9079959852793577, 0.9079959852793577]
acc_fs_stability_adv_Heart_Disease = [0.78, 0.8, 0.7997351870241642, 0.835493519441675, 0.8711943793911008, 0.8711943793911008]
acc_fs_rf_avg_Heart_Disease = [0.7921052631578948, 0.7921052631578946, 0.7868421052631579, 0.7447368421052631, 0.7105263157894737, 0.5394736842105263]
acc_fs_rf_adv_avg_Heart_Disease = [0.7921052631578948, 0.8105263157894737, 0.8210526315789473, 0.8105263157894737, 0.7447368421052631, 0.6710526315789473]


acc_fs_stability_Ionosphere = [0.46166666666666656, 0.3976757369614512, 0.2696784922394678, 0.35049019607843135, 0.41594136509390756, 0.4624505928853754]
acc_fs_stability_adv_Ionosphere = [0.46166666666666656, 0.4049999999999999, 0.2941176470588234, 0.2917666996186977, 0.2979278365454281, 0.3272102747909199]
acc_fs_rf_avg_Ionosphere = [0.915909090909091, 0.934090909090909, 0.9227272727272726, 0.9295454545454547, 0.9204545454545453, 0.9136363636363635]
acc_fs_rf_adv_avg_Ionosphere = [0.915909090909091, 0.934090909090909, 0.9340909090909092, 0.9386363636363637, 0.9295454545454545, 0.9409090909090908]

acc_fs_stability_name_list = [acc_fs_stability_scikit_breast, acc_fs_stability_Parkinson, acc_fs_stability_Heart_Disease, acc_fs_stability_Ionosphere]
acc_fs_stability_adv_name_list = [acc_fs_stability_adv_scikit_breast, acc_fs_stability_adv_Parkinson, acc_fs_stability_adv_Heart_Disease, acc_fs_stability_adv_Ionosphere]
acc_rf_avg_name_list = [acc_fs_rf_avg_scikit_breast, acc_fs_rf_avg_Parkinson, acc_fs_rf_avg_Heart_Disease, acc_fs_rf_avg_Ionosphere]
acc_rf_adv_avg_name_list = [acc_fs_rf_adv_avg_scikit_breast, acc_fs_rf_adv_avg_Parkinson, acc_fs_rf_adv_avg_Heart_Disease, acc_fs_rf_adv_avg_Ionosphere]

for l in acc_fs_stability_name_list:
    acc_fs_stability.append(statistics.mean(l))
print(acc_fs_stability)

for l in acc_fs_stability_adv_name_list:
    acc_fs_stability_adv.append(statistics.mean(l))
print(acc_fs_stability_adv)

for l in acc_rf_avg_name_list:
    acc_fs_rf_avg.append(statistics.mean(l))
print(acc_fs_rf_avg)

for l in acc_rf_adv_avg_name_list:
    acc_fs_rf_adv_avg.append(statistics.mean(l))
print(acc_fs_rf_adv_avg)

