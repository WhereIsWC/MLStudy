from numpy import *
#加载测试数据
def loadDataSet():
    dataMat = []; labelMat = []
    fr = open("regres/testSet.txt")
    for line in fr.readlines():
        lineArr = line.strip().split() #删除头尾空格，并按照空格分割字符
        dataMat.append([10., float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1/(1+exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    alpha = 0.001;maxCycles = 500;weights = ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat-h)
        weights = weights +alpha * dataMatrix.transpose()* error
    return weights
    #81页