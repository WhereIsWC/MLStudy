from distutils.log import error
from numpy import *


#加载测试数据
def loadDataSet():
    dataMat = []; labelMat = []
    fr = open("regres/testSet.txt")
    for line in fr.readlines():
        lineArr = line.strip().split() #删除头尾空格，并按照空格分割字符
        dataMat.append([1, float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1/(1+exp(-inX))

# 梯度上升优化算法
def gradAscent(dataMatIn, classLabels):
    #转换为nupmy的矩阵
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    alpha = 0.001  #向目标移动的步长
    maxCycles = 500  # 迭代次数
    weights = ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat-h)
        weights = weights +alpha * dataMatrix.transpose() * error
    return weights

#随机梯度上升算法 numIter代表迭代次数
def stocGradAscent0(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1+j+i)+0.01 #使alpha在每次迭代都在变换，但永远不会等于0，常数可以调整
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha  * dataMatrix[randIndex] * error
            del(dataIndex[randIndex])
    return weights


#画出决策边界
def plotBestFit(wei):
    import matplotlib.pyplot as plt
    if type(wei).__name__ == "matrix":
        weights =wei.getA()
    else:
        weights = wei
    dataMat,labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c="red",marker="s")
    ax.scatter(xcord2,ycord2,s=30,c="green")
    x = arange(-3,3,0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel("X1");plt.ylabel("X2")
    plt.show()
    a = 1 # 调试用，无意义


#回归分类函数
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1
    else:
        return 0


#测试函数
def colicTest():
    frTrain = open("regres/horseColicTraining.txt")
    frTest = open("regres/horseColicTest.txt")
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split("\t")
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent0(array(trainingSet),trainingLabels,500)
    errorCount = 0;numTestVec = 0
    for line in frTest.readlines():
        numTestVec += 1
        currLine = line.strip().split("\t")
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr),trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate

def multTest():
    numTests = 10; errorSum = 0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests,errorSum/float(numTests)))

multTest()

# dataArr, labelMat =loadDataSet()
# weights=stocGradAscent0(array(dataArr),labelMat)
# #weights=gradAscent(dataArr,labelMat)

# plotBestFit(weights)