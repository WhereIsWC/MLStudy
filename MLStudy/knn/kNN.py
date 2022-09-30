from cProfile import label
from tokenize import group
from numpy import *
#导入操作符模块
import operator
import matplotlib
import matplotlib.pyplot as plt

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

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


datingDataMat,datingLabels = file2matrix("D:/hst/MLStudy/MLStudy/knn/datingTestSet.txt")
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))
plt.show()