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

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not my Vocabulary" % word)
    return returnVec

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num=p1Num = zeros(numWords)
    p0Denom = p1Denom = 0
    for i in range(numTrainDocs):
        if trainCategory[i]:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num/p1Denom
    p0Vect = p0Num/p0Denom
    return p0Vect,p1Vect,pAbusive

listOPosts, listClasses = loadDataSet()
myVocabList = createVocabList(listOPosts)

trainMat = []
for postinDoc in listOPosts:
    trainMat.append(setOfWords2Vec(myVocabList,postinDoc))

p0v,p1v, pAb = trainNB0(trainMat,listClasses)

print(p0v,p1v, pAb)