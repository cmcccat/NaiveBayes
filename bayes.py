from numpy import *
import re
def loadDataSet():
    postingList = [['my','dog','has','flea','problems','help','please'],
                   ['maybe','not','take','him','to','dog','park','stupid'],
                   ['my','dalmation','is','so','cute','I','love','him'],
                   ['stop','posting','stupid','worthless','garbage'],
                   ['mr','licks','ate','my','steak','how','to','stop','him'],
                   ['quit','buying','worthless','dog','food','stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList,classVec

#返回一个不重复的分词集合（词汇表）
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        #创建两个集合的并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

#输入词汇表和文档，返回一个分词向量
def setOfWords2Vec(vocavList,inputSet):
    returnVec = [0]*len(vocavList)
    for word in inputSet:
        if word in vocavList:
            returnVec[vocavList.index(word)] = 1
        else: print("the word: {} is not in my Vocabulary!".format(word))
    return returnVec

#输入文档矩阵和每篇文档所构成的
def trainNB0(trainMatrix,trainCategory):
    #训练文档数量
    numTrainDocs = len(trainMatrix)
    #词汇向量表
    numWords = len(trainMatrix[0])
    #标记正样本概率
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        #如果为标记文档,该类别词汇量/该类别总词汇量
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    return  p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClassl):
    p1 = sum(vec2Classify * p1Vec) + log(pClassl)
    p0 = sum(vec2Classify * p0Vec) + log(1.0-pClassl)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    #读取数据和标签
    listOPosts,listClasses = loadDataSet()
    #创建词汇表
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    #创建词汇
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love','my','dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid','garbage']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb))

#返回文档中出现的单词数
def bagOfWorfs2VecMN(vocabList,inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

#数据文档分割
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W+',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1,26):
        #获取文档单词
        wordList = textParse(open('email/spam/{}.txt'.format(i)).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/{}.txt'.format(i)).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    #创建词典
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):
        #生成随机测试数据集索引
        randIndex = int(random.uniform(0,len(trainingSet)))
        # 如果随机数已经存在，可能需要进一步处理
        while randIndex in testSet:
            randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses =[]
    #构建训练数据集
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    #测试数据
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            #print(docList[docIndex])
    print("the error rate is:", float(errorCount)/len(testSet))

#获得词频前30的词语
def calcMostFreq(vocabList,fullText):
    freqDict = {}
    for token in vocabList:
         #统计词频
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(),key = lambda item : item[1] ,reverse = True)
    return sortedFreq[:30]

def localWords(feed1,feed0):
    import feedparser
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        #分割成单词向量
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    #获得词汇表
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList,fullText)
    #去除高频词
    for pairW in top30Words:
        if pairW[0] in vocabList:vocabList.remove(pairW[0])
    trainingSet = list(range(2*minLen))
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWorfs2VecMN(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWorfs2VecMN(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is:',float(errorCount)/len(testSet))
    return vocabList,p0V,p1V

#获得高频词汇
def getTopWords(ny,sf):
    vocabList,p0V,p1V = localWords(ny,sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -5.0 :
            topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -5.0 :
            topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF,key = lambda  pair:pair[1],reverse = True)
    print('SF**SF**SF**SF**SF')
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY,key = lambda pair:pair[1],reverse= True)
    print('NY**NY**NY**Ny**NY')
    for item in sortedNY:
        print(item[0])