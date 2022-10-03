from cProfile import label
from doctest import Example
from math import log
import operator

#计算数据集的熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

#创建测试数据
def createDate():
    dataSet = [[1,1,"yes"],
                [1,1,"yes"],
                [1,0,"no"],
                [0,1,"no"],
                [0,1,"no"]]
    labels = ["no surfacing","flippers"]
    return dataSet,labels

#按照给定特征划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    #因为最后一列是标签列，所以-1
    numFeatures = len(dataSet[0])- 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0
    bestFeature = -1
    for i in range(numFeatures):
        #取第i列的所有数据，并转换成1行
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain> bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]

#创建数树
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    #类别完全相同则停止继续划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #当遍历完，即只有一组数据时
    if len(dataSet[0]) == 1:
        return  majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    #删除当前已经判断过的标签
    del(labels[bestFeat])
    #把这特征值下面的所有数据取出来
    featValues = [example[bestFeat] for example in dataSet]
    #去重
    uniqueVals = set(featValues)
    #对不同的值递归调用本方法，并写入myTree中
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree

#使用决策树的分类函数
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == "dict":
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def storeTree(inputTree, fileName):
    import pickle
    fw = open(fileName,"wb")
    pickle.dump(inputTree,fw)
    fw.close

def grabTree(fileName):
    import pickle
    fr = open(fileName,"rb")
    return pickle.load(fr)




myDat,labels = createDate()
featLabels = list(labels)
myTree = createTree(myDat, labels)
print(myTree)
print(classify(myTree,featLabels,[1,0]))
print(classify(myTree,featLabels,[1,1]))
storeTree(myTree,"StoreTreeTest.txt")
print(grabTree("StoreTreeTest.txt"))
