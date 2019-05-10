import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'C:/Users/Serhan/Documents/SLACwork/VADER-Analytics/mlpowerflow')
import mlpf

def removeValues(data, percentage, inplace = False):
    """
    Removes individual values in a pandas dataframe with probabilty percentage
    """

    if inplace:
        dataRemoved = data
    else:
        dataRemoved = data.copy()

    numRows, numCols = data.shape
    for i in range(numCols):
        for j in range(numRows):
            if np.random.uniform() < percentage:
                dataRemoved[i][j] = None


    return dataRemoved

def removeRows(data, rowPercentage, colPercentage = 1, inplace = False):
    """
    Selects rows with probability rowPercentage, then for each column removes that column
    from that row with probability ColPercentage
    """

    if inplace:
        dataRemoved = data
    else:
        dataRemoved = data.copy()

    numRows, numCols = data.shape
    for j in range(numRows):
        if np.random.uniform() < rowPercentage:
            for i in numCols:
                if np.random.uniform() < colPercentage:
                    dataRemoved[i][j] = None

    return dataRemoved


def nonNullIntersection(dataList,):
    """
    Takes a list of dataframes and returns the list of dataframes with rows that contain a null value in
    any dataframe removed. This is so that the row can be used for training ML Powerflow
    """
    dataList = dataList.deepcopy()

    masks = []
    for data in dataList:
        mask = data.isna().any(axis = 1)
        masks.append(mask)

    finalMask = masks[0]
    for mask in masks:
        finalMask = finalMask & mask

    for x in range(len(dataList)):
        dataList[x] = dataList[x][finalMask]

    return dataList

def fillValuesMLPFForward(p, q, v, a):
    """
    Takes real and reactive power and voltage and phase angle and trains
    an mlpowerflow model to fill in missing power data
    """
    num_samples, num_bus = p.shape

    model = mlpf.ForwardMLPF(num_bus, num_samples)

    dataNonNull = nonNullIntersection([p,q,v,a].deepcopy())
    pNonNull = dataNonNull[0].values
    qNonNull = dataNonNull[1].values
    vNonNull = dataNonNull[2].values
    aNonNull = dataNonNull[3].values

    model.supply_full_data(pNonNull, qNonNull, vNonNull, aNonNull)

    model.train_test_split_data(train_percent = 1)
    C_set = [0.01, 0.1, 2.0]
    eps_set = [1e-7, 1e-5, 1e-3]
    max_iter = 1e3
    model.fit(C_set, eps_set, max_iter)

    voltage = pd.concat([v,a])
    power = pd.concat([p,q])

    for j in range(num_samples):
        if np.isnan(np.sum(power[j])):
            X = voltage[j]
            predictions = model.apply_svr(X)
            for i in range(num_bus):
                if np.isnan(power[j,i]):
                    power[j,i] = predictions[i]

    pFilled = pd.DataFrame(power[:, :num_bus])
    qFilled = pd.DataFrame(power[:, num_bus:])

    return pFilled, qFilled


