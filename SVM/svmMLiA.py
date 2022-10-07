from cProfile import label
import os
from random import random
from numpy import *

def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split("\t")
        dataMat.append((float(lineArr[0]),float(lineArr[1])))
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat


def selectJrand(i,m):
    j = i 
    while(j==i):
        j = int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

"""
dataMatIn:数据集，classLabels类别标签
C：常数
toler：容错率
maxIter：取消前最大的循环次数
"""
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()
    b = 0; m,n = shape(dataMatrix)
    alphas = mat(zeros((m,1)))
    iter = 0
    while(iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) +b
            Ei =fXi - float(labelMat[i])
            # 判断alpha是否能够优化
            if((labelMat[i]*Ei < -toler) and (alphas[i]<C)) or ((labelMat[i]*Ei > toler)
            and (alphas[i]> 0 )):
                j = selectJrand(i,m) #随机选择第二个alpha
                fXj = float(multiply(alphas,labelMat).T *(dataMatrix*dataMatrix[j,:].T))+b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # 保证alpha在0到C之间
                if (labelMat[i] != labelMat[j]):
                    L = max(0,alphas[j] - alphas[i])
                    H = min(C,C+alphas[i] - alphas[j])
                else:
                    L = max(0,alphas[j] + alphas[i] - C)
                    H = min(C,alphas[i] + alphas[j])        
                if L == H:
                    print("L==H")
                    continue
                #eta是alpha[j]的最优修改量
                eta =2 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T
                - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0:
                    print("eta>=0")
                    continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if(abs(alphas[j]- alphaJold) < 0.00001):
                    print("j not moving enough")
                    continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j]) #对i进行修改，修改量与j相同，方向相反
                b1 = b -Ei -labelMat[i]*(alphas[i] - alphaIold)* dataMatrix[i,:]*dataMatrix[i,:].T 
                - labelMat[j]*(alphas[j] - alphaJold) * dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i,:]*dataMatrix[j,:].T  
                - labelMat[j]*(alphas[j]- alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if(0<alphas[i]) and (C> alphas[i]):
                    b = b1
                elif(0< alphas[j] ) and (C > alphas[j]):
                    b = b2
                else:
                    b=(b1 + b2)/2
                alphaPairsChanged += 1
                print("iter: %d,i: %d ,pairs changed %d" % (iter, i, alphaPairsChanged))
            if(alphaPairsChanged==0):
                iter += 1
            else:
                iter = 0
            print("iteration number : %d" % iter)
    return b,alphas

#定义一个数据结构来缓存数据
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler,kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0 
        self.eCache = mat(zeros((self.m,2))) # 误差缓存
        self.K = mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:],kTup)

#计算误差值
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:,k] +oS.b)
    Ek = fXk -float(oS.labelMat[k])
    return Ek

# 选择第二个alpha
def selectJ(i, oS, Ei):
    maxK = -1;maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]
    validEchaeList = nonzero(oS.eCache[:,0].A)[0]
    if(len(validEchaeList)>1):
        for k in validEchaeList:
            if k == i:
                continue
        Ek = calcEk(oS,k)
        deltaE = abs(Ei - Ek)
        # 选择具有最大步长的j
        if(deltaE > maxDeltaE):
            maxK = k; maxDeltaE = deltaE; Ej =Ek
        return maxK,Ej
    else:
        j = selectJrand(i,oS.m)
        Ej = calcEk(oS , j)
    return j, Ej


#更新误差值
def updateEk(oS,k):
    Ek = calcEk(oS , k)
    oS.eCache[k] = [1,Ek]


def innerL(i, oS):
    Ei = calcEk(oS, i)
    if((oS.labelMat[i]* Ei < -oS.tol) and (oS.alphas[i] < oS.C)) \
    or ((oS.labelMat[i] * Ei >oS.tol) and (oS.alphas[i] > 0)):
        j,Ej =selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy();alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C,oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L==H")
            return 0
        eta = 2 * oS.K[i,j] - oS.K[i,i] -oS.K[j,j]
        if eta >= 0:
            print("eta >= 0")
            return 0
        oS.alphas[j] -= oS.labelMat[j]* (Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS,j)
        if(abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j]* oS.labelMat[i]*(alphaJold - oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b -Ej -oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i,i]  \
            - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i,j]
        b2 = oS.b -Ej -oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i,j]  \
            - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j,j]
        if(0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b= b1
        elif(0< oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2)/2
        return 1
    else: return 0 


def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=("lin",0)):
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler,kTup)
    iter = 0 
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)
            print("fullSet, iter: %d i: %d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print(" non-bound, iter: %d i %d, pairs changed %d" %(iter, i ,alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b,oS.alphas
 

def calcWs(alphas, dataArr, classLabels):
    X = mat(dataArr); labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w =zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

#核函数
def kernelTrans(X, A, kTup):
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if( kTup == "lin"):
        K = X * A.T
    elif kTup[0] == "rbf":
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2))
    else:
        raise NameError("HeSongtao We Have a Problem -- That Kernel is not recognized")
    return K


def testRbf(k1 = 1.3):
    dataArr,labelArr =loadDataSet("SVM/testSetRBF.txt")
    b,alphas = smoP(dataArr,labelArr,200,0.0001 ,10000, ("rbf",k1))
    dataMat = mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A>0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d support vectors" % shape(sVs)[0])
    m,n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,dataMat[i,:],("rbf",k1))
        predict = kernelEval.T * multiply(labelSV,alphas[svInd]) +b
        if (sign(predict) != sign(labelArr[i])):
            errorCount += 1
    print("the training error rate is : %f" % (float(errorCount/m)))
    dataArr,labelArr = loadDataSet("SVM/testSetRBF2.txt")
    errorCount = 0
    dataMat = mat(dataArr);labelMat = mat(labelArr).transpose()
    m,n = shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,dataMat[i,:],("rbf",k1))
        predict = kernelEval.T * multiply(labelSV,alphas[svInd])+ b
        if (sign(predict) != sign(labelArr[i])):
            errorCount += 1
    print("the test error rate is :%f" % float(errorCount/m))

testRbf()

# dataArr, labelArr = loadDataSet("SVM/testSet.txt")
# # smoSimple
# b,alphas = smoP(dataArr,labelArr,0.6,0.001,40)
# #b,alphas = smoSimple(dataArr,labelArr,0.6,0.001,40)
# ws = calcWs(alphas, dataArr, labelArr)
# dataMat = mat(dataArr)
# for i in range(len(dataMat)):
#     if (dataMat[i]*mat(ws) + b) > 0 and labelArr[i] < 1:
#         print(dataMat[i]*mat(ws) + b,labelArr[i])
#     elif (dataMat[i]*mat(ws) + b) <= 0 and labelArr[i] > -1:
#         print(dataMat[i]*mat(ws) + b,labelArr[i])
