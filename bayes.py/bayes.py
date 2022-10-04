from numpy import *

def loadDataSet():
    postingList=[["my","dog","has","flea","problems","help","please"],
                ["maybe","not","take","him","to","dog","park","stupid"],
                ["my","dalmation","is","so","cute","i","love","him"],
                ["stop","posting","stupid","worthless","garbage"],
                ["mr","licks","ate","my","steak","how","to","stop","him"],
                ["quit","buying","worthless","dog","food","stupid"]]
    classVec = [0,1,0,1,0,1]#1带有侮辱文字，0代表正常
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)
#词集模型，此模型不能反应某个词多次出现的情况
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not my Vocabulary" % word)
    return returnVec

#词袋模型，解决不能反应某个词多次出现的问题
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    #p0Num=p1Num = zeros(numWords)
    #p0Denom = p1Denom = 0
    #此处做如下修改是为了避免乘数是0，导致无论其他数是多少结果都为0的情况
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2
    p1Denom = 2
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # p1Vect = p1Num/p1Denom
    # p0Vect = p0Num/p0Denom
    # 此处做如下修改是为了处理数太小,小数点溢出的问题
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive


#分类函数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify*p1Vec) + log(pClass1)
    p0 = sum(vec2Classify*p0Vec) + log(1-pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0v,p1v,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ["love","my","dalmation"]
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,"classified as: " ,classifyNB(thisDoc,p0v,p1v,pAb))
    testEntry = ["stupid","garbage"]
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,"classified as: " ,classifyNB(thisDoc,p0v,p1v,pAb))

# 将文本分割为词组成的列表，目前规则：剔除小于3位数的单词，并转换为小写 
def textParse(bigString):
    import re
    listOfTokens = re.split(r"\W* ", bigString) #使用正则表达式分割语句
    return [tok.lower() for tok in listOfTokens if len(tok)>2] #剔除小于3位数的词，并转换为小写

def spamTest():
    docList = []; classList = []; fullText = []
    for i in range(1,26):
        wordList = textParse(open("bayes.py/email/spam/%d.txt" % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open("bayes.py/email/ham/%d.txt" % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50));testSet = []
    # 随机取10个作为测试集，40个作为训练集
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0v,p1v,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0v,p1v,pSpam) != classList[docIndex]:
            errorCount += 1
            print(docList[docIndex])
    print("the error rate is:" , float(errorCount)/len(testSet))

spamTest()

# listOPosts, listClasses = loadDataSet()
# myVocabList = createVocabList(listOPosts)

# trainMat = []
# for postinDoc in listOPosts:
#     trainMat.append(setOfWords2Vec(myVocabList,postinDoc))

# p0v,p1v, pAb = trainNB0(trainMat,listClasses)
# print(p0v,p1v, pAb)