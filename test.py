import bayes
import  re
import feedparser
'''
listOPosts,listClasses = bayes.loadDataSet()
myVocablist = sorted(bayes.createVocabList(listOPosts))
print(myVocablist)

trainMat = []
for postinDoc in listOPosts:
    trainMat.append(bayes.setOfWords2Vec(myVocablist,postinDoc))
p0V,p1V,pAb=bayes.trainNB0(trainMat,listClasses)
print(pAb)
print(p0V)
print(p1V)

bayes.testingNB()
regEx = re.compile('\\W+')
emailText = open('email/ham/6.txt').read()
listOfTokens = regEx.split(emailText)
result = [tok.lower() for tok in listOfTokens if len(tok) > 0]
print(result)
bayes.spamTest()
'''
ny = feedparser.parse('https://newyork.craigslist.org/stp/index.rss')
sf = feedparser.parse('https://sfbay.craigslist.org/stp/index.rss')
#vocabList,pSF,pNY = bayes.localWords(ny,sf)
print (ny)
print (sf)
bayes.getTopWords(ny,sf)