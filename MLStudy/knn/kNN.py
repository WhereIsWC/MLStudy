from cProfile import label
from tokenize import group
from numpy import *
#导入操作符模块
import operator
import matplotlib
import matplotlib.pyplot as plt

#创建测试数据
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

#分类器
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    #计算距离
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    #选择距离最小的k个点
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    #距离由近到远排序
    sortedClassCount = sorted(classCount. items() ,
    key=operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]


#将N行4列文件的前三列转换为N行3列的矩阵，最后一列转换为标签字典
def file2matrix(filename):
    fr = open(filename)
    #读取文件，并返回行数
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    #创建文件行数*3的空矩阵，此处的3需要根据需求变动
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    #循环将文件内容写入矩阵中
    for line in arrayOfLines:
        #删除每一行中头尾的空格
        line  = line.strip()
        listFromLIne = line.split("\t")
        returnMat[index,:] = listFromLIne[0:3]
        classLabelVector.append(int(listFromLIne[-1]))
        index += 1
    return returnMat,classLabelVector 

#将数组中的值归一化为特征值，避免因为量纲差异带来的误差
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

#测试模型
def datingClassTest():
    # 测试数据占比
    horatio = 0.10
    datingDataMat,datingLabels = file2matrix("D:/hst/MLStudy/MLStudy/knn/datingTestSet.txt")
    # 将测试数据标准化
    normMat , ranges ,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*horatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("判断结果为%d，实际结果为%d" %(classifierResult,datingLabels[i]))
        if(classifierResult!=datingLabels[i]):
            errorCount += 1.0
    print("错误率为：%f"% float(errorCount/numTestVecs))


#开始预测
def classifyPerson():
    resultList = ["not at all","in small doses","in large doses"]
    percentTats = float(input("每周游戏上花费时间："))
    iceCream = float(input("每周吃多少冰淇淋："))
    ffMiles = float(input("今年飞行里程数"))
    inArr = array([percentTats,ffMiles,iceCream])
    datingDataMat,datingLabels = file2matrix("D:/hst/MLStudy/MLStudy/knn/datingTestSet.txt")
    normMat , ranges ,minVals = autoNorm(datingDataMat)
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print("You will probably like this person: ",resultList[classifierResult-1])

classifyPerson()
"""
#画散点图展示数据部分
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))
ax.set_xlabel("I am x")
ax.set_ylabel("I am y")
plt.show()
"""
