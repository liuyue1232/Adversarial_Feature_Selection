import sklearn
from sklearn import metrics
import scipy.io as sio
import math
import numpy as np

def getMutualInfos(data, labels):
    '''
    This function takes as input the data and labels and returns the mutual information of each feature
    with the labels in a np.dnarray of length d

    INPUTS:
    - data is a 2-dimensional numpy.ndarray where rows are examples and columns are features
    - labels is a 1-dimansional numpy.ndarray giving the label of each example in data

    OUPUT:
    - a 1-dimensional numpy.ndarray of length d (where d is the number of features)
      with the mutual information of each feature with the label
    '''
    M, d = data.shape
    mutualInfos = np.zeros(d)
    # for each feature
    for f in range(d):
        # we calculate the mutual information of the feature with the labels
        mutualInfos[f] = metrics.mutual_info_score(data[:, f], labels)
        print(mutualInfos[f])
    return mutualInfos


def getBootstrapSample(data, labels):
    '''
    This function takes as input the data and labels and returns
    a bootstrap sample of the data, as well as its out-of-bag (OOB) data

    INPUTS:
    - data is a 2-dimensional numpy.ndarray where rows are examples and columns are features
    - labels is a 1-dimansional numpy.ndarray giving the label of each example in data

    OUPUT:
    - a dictionnary where:
          - key 'bootData' gives a 2-dimensional numpy.ndarray which is a bootstrap sample of data
          - key 'bootLabels' is a 1-dimansional numpy.ndarray giving the label of each example in bootData
          - key 'OOBData' gives a 2-dimensional numpy.ndarray the OOB examples
          - key 'OOBLabels' is a 1-dimansional numpy.ndarray giving the label of each example in OOBData
    '''
    m, d = data.shape
    if m != len(labels):
        raise ValueError('The data and labels should have a same number of rows.')
    ind = np.random.choice(range(m), size=m, replace=True)
    OOBind = np.setdiff1d(range(m), ind, assume_unique=True)
    bootData = data[ind,]
    bootLabels = labels[ind]
    OOBData = data[OOBind,]
    OOBLabels = labels[OOBind]
    return {'bootData': bootData, 'bootLabels': bootLabels, 'OOBData': OOBData, 'OOBLabels': OOBLabels}


def generateAtificialDataset(m, d, d_rel, rho):
    ''' This function enerates the artificial dataset used in the experiments (Section 7.1) of [1].
        The data set is made of continuous data where the first
        d_rel featues are relevant and where the d-d_rel remaining features are irrelevant to the target class
        this is a balanced data set where m/2 examples belong to class -1 and m/2 to class 1

        INPUTS:
        m is the number of samples
        d is the number of features/variables
        d_rel is the number of relevant features
        rho is the degree of redundancy (should be between 0 and 1)

        OUPUTS:
        A dictionnary with the data and the labels
    '''
    if d_rel >= d:
        raise ValueError(
            'The input number of relevant features d_rel must be strictly less than the total number of features d')
    if rho < 0 or rho > 1:
        raise ValueError(
            'The input argument rho controlling the degree of redundancy between the relevant features must be a value between 0 and 1.');
    num_positives = int(m / 2)  ## Take half instances as positive examples
    num_negatives = m - num_positives
    labels = np.concatenate((np.ones((num_positives), dtype=np.int8), -np.ones((num_negatives), dtype=np.int8)))
    mu_plus = np.concatenate((np.ones((d_rel), dtype=np.int8), np.zeros((d - d_rel))))  ## mean of the positive examples
    mu_minus = np.concatenate((-np.ones((d_rel), dtype=np.int8), np.zeros(d - d_rel)))  ## mean of the negative examples
    Sigma_star = rho * np.ones((d_rel, d_rel), dtype=np.int8) + (1 - rho) * np.eye(d_rel)
    sub1 = np.concatenate((Sigma_star, np.zeros((d_rel, d - d_rel))))
    sub2 = np.concatenate((np.zeros((d - d_rel, d_rel)), np.eye(d - d_rel)))
    Sigma = np.concatenate((sub1, sub2), axis=1)  ## the covariance matrix
    positive_ex = np.random.multivariate_normal(mu_plus, Sigma, num_positives)
    negative_ex = np.random.multivariate_normal(mu_minus, Sigma, num_negatives)
    data = np.concatenate((positive_ex, negative_ex), axis=0)
    ## we randomly permute the examples...
    order = ind = np.random.choice(range(m), size=m, replace=False)
    data = data[order,]
    labels = labels[order]
    trueRelevantSet = np.zeros(d)
    trueRelevantSet[range(d_rel)] = 1
    return {'data': data, 'labels': labels, 'trueRelevantSet': trueRelevantSet}
