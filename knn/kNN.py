from cProfile import label
from tokenize import group
from numpy import *
from os import listdir
from PIL import Image
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
        listFromLIne = line.split("/t")
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

#classifyPerson()
"""
#画散点图展示数据部分
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))
ax.set_xlabel("I am x")
ax.set_ylabel("I am y")
plt.show()
"""

#以下是手写数字识别部分


#将2进制（32*32）存储的图片转换为矩阵（1*1024）
#其中的32是图片的总行数和列数，实际应用中要么图片适应代码，要么修改代码
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i  in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def arr2vector(arrUnder2):
    returnVect = zeros((1,1024))
    for i  in range(32):
        for j in range(32):
            returnVect[0,32*i+j] = int(arrUnder2[i,j])
    return returnVect

def handWritingClassTest():
    hwLabels = []
    # 获取训练数据目录下的文件名称
    trainingFileList = listdir("kNN/trainingDigits")
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    #生成训练集及其标签
    for i in range(m):
        fileNameStr = trainingFileList[i]
        #获取不带.txt的文件名
        fileStr = fileNameStr.split(".")[0]
        #获取_前的文本并强制转换为数字形式，这是因为我们在处理文件时使用了对应"答案_序号.txt"的命名规则
        classNumStr = int(fileStr.split("_")[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector("knn/trainingDigits/%s" % fileNameStr)
    testFileList = listdir("knn/testDigits")
    errorCount = 0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumStr = int (fileStr.split("_")[0])
        vectorUnderTest  = img2vector("knn/testDigits/%s" % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels , 3)
        print("分类器返回%d,正确答案为%d" % (classifierResult,classNumStr))
        if(classifierResult!=classNumStr): errorCount += 1
    print("总测试样本为%d,错误个数为%d,错误率为%f" %(mTest,errorCount,errorCount/mTest))   

#将传入的图片转换为0,1的32*32的数组    
def pic2Array(fileName):
    img = Image.open(fileName)
    img = img.resize((32,32))
    #转换为灰度图片
    img = img.convert("L")
    img_new = img.point(lambda x:0 if x>170 else 1)
    returnArr = array(img_new)
    """
    with open("d:/test.txt") as f:
        for i in range(32):
            for j in range(32):
                f.write(str(returnArr[i,j]))
            f.write("\n")    
    """
    return returnArr

#自己手写图片识别测试，算法问题不大，主要问题在于图片转换成0,1矩阵过程中失真严重，导致识别错误。后期再来优化
def handWritingClassTest1(fileName):
    hwLabels = []
    # 获取训练数据目录下的文件名称
    trainingFileList = listdir("kNN/trainingDigits")
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    #生成训练集及其标签
    for i in range(m):
        fileNameStr = trainingFileList[i]
        #获取不带.txt的文件名
        fileStr = fileNameStr.split(".")[0]
        #获取_前的文本并强制转换为数字形式，这是因为我们在处理文件时使用了对应"答案_序号.txt"的命名规则
        classNumStr = int(fileStr.split("_")[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector("knn/trainingDigits/%s" % fileNameStr)


    arrayUnderTest = pic2Array(fileName)

    vectorUnderTest  = arr2vector(arrayUnderTest)
    classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels , 3)
    print("分类器返回%d" % classifierResult)

handWritingClassTest1("d:/6.png")