import pandas as pd
import numpy as np

dataset =pd.read_csv("movie.csv")
dataset.head()
item =pd.read_csv("item.csv",encoding = "ISO-8859-1", index_col = False)

dataset = pd.merge(dataset, item,left_on='itemID',right_on='itemID')

userID = dataset.userID
userID2 = item[['itemID']]
dataset.loc[0:10,['userID']]

numUsers = max(dataset.userID)
numMovies = max(dataset.itemID)

#moviesPerUser = dataset.userID.value.counts()
#userPerMovie = dataset.title.value_counts()
def HighestRated(activeUser, N):
  topMovies = pd.DataFrame.sort_values(dataset[dataset.userID == activeUser], ['rating'], ascending=[0])[:N]
  return list(topMovies.title)
HighestRated(5,5)
 
userRatingMatrix = pd.pivot_table(dataset, values = 'rating', index = ['userID'], columns = ['itemID'])

from scipy.spatial.distance import correlation
def UserSimilarity(user1, user2):
    user1 = np.array(user1)-np.nanmean(user1)
    user2 = np.array(user2)-np.nanmean(user2)
    CommonIds = [i for i in range(len(user1))
                if user1[i]>0 and user2[i]>0]
    if len(CommonIds)==0:
        return 0
    else:
        user1 = np.array([user1[i] for i in CommonIds])
        user2 = np.array([user2[i] for i in CommonIds])
        return correlation(user1,user2)
    
    
def nearestneighborRating(activeUser, K):
    similarityMatrix=pd.DataFrame(index=userRatingMatrix.index, columns=['Similarity'])
    for i in userRatingMatrix.index:
        similarityMatrix.loc[i]=UserSimilarity(userRatingMatrix.loc[activeUser],userRatingMatrix.loc[i])
        similarityMatrix = pd.DataFrame.sort_values(similarityMatrix,['Similarity'], ascending=[0])
        nearestneighbors = similarityMatrix[:K]
        neighborItemRating=userRatingMatrix.loc[nearestneighbors.index]
        predictRating= pd.DataFrame(index = userRatingMatrix.columns, columns=['Ratings'])
        for i in userRatingMatrix.columns:
            predictedRating= np.nanmean(userRatingMatrix.loc[activeUser])
            for j in neighborItemRating.index:
                    if userRatingMatrix.loc[j,i]>0:
                        predictedRating+= (userRatingMatrix.loc[j,i]-np.nanmean
                                           (userRatingMatrix.loc[j]))*nearestneighbors.loc[j,'Similarity']
        predictRating.loc[i,'Rating']=predictedRating
    return predictedRating 


def topNRecommendations(activeUser,N):
    predictedRating = nearestneighborRating(activeUser,10)
    moviesUserAlreadyWatched = list(userRatingMatrix.loc[activeUser].loc
                                    [userRatingMatrix.loc[activeUser]>0].index)
    predictedRating = predictedRating.drop(moviesUserAlreadyWatched)
    topRecommendations = pd.DataFrame.sort_values(predictedRating
                                                  ['Rating'],ascending = [0])[:N]
    topRecommendationsTitles = (item.loc[item.itemID.isin(topRecommendations.index)])
    return list(topRecommendationsTitles.titles)

activeUser = 5
result = HighestRated(activeUser,5),"\n",topNRecommendations(activeUser,3)

def matrixFactorization(R, K, steps=10, gamma=0.001,lamda=0.02):
    N=len(R.index)# Number of users
    M=len(R.columns) # Number of items 
    P=pd.DataFrame(np.random.rand(N,K),index=R.index)
    Q=pd.DataFrame(np.random.rand(M,K),index=R.columns)
    for step in range(steps):
        for i in R.index:
            for j in R.columns:
                if R.loc[i,j]>0:
                    eij=R.loc[i,j]-np.dot(P.loc[i],Q.loc[j])
                    P.loc[i]=P.loc[i]+gamma*(eij*Q.loc[j]-lamda*P.loc[i])
                    Q.loc[j]=Q.loc[j]+gamma*(eij*P.loc[i]-lamda*Q.loc[j])
        e=0
        for i in R.index:
            for j in R.columns:
                if R.loc[i,j]>0:
                    e= e + pow(R.loc[i,j]-np.dot(P.loc[i],Q.loc[j]),2)+lamda*(pow(np.linalg.norm(P.loc[i]),2)+pow(np.linalg.norm(Q.loc[j]),2))
        if e<0.001:
            break
        step
    return P,Q 
(P,Q)=matrixFactorization(userRatingMatrix.iloc[:100,:100],K=2,gamma=0.001,lamda=0.02, steps=100)

activeUser=1
predictItemRating=pd.DataFrame(np.dot(P.loc[activeUser],Q.T),index=Q.index,columns=['Rating'])
topRecommendations=pd.DataFrame.sort_values(predictItemRating,['Rating'],ascending=[0])[:3]
topRecommendationTitles=item.loc[item.itemID.isin(topRecommendations.index)]
list(topRecommendationTitles.title)

import itertools 
allitems=[]

def support(itemset):
    userList=userRatingMatrix.index
    nUsers=len(userList)
    ratingMatrix=userRatingMatrix
    for item in itemset:
        ratingMatrix=ratingMatrix.loc[ratingMatrix.loc[:,item]>0]
        userList=ratingMatrix.index
    return float(len(userList))/float(nUsers)

minsupport=0.3
for item in list(userRatingMatrix.columns):
    itemset=[item]
    if support(itemset)>minsupport:
        allitems.append(item)
len(allitems)
minconfidence=0.1
assocRules=[]
i=2
for rule in itertools.permutations(allitems,2):
    #Generates all possible permutations of 2 items from the remaining
    # list of 47 movies 
    from_item=[rule[0]]
    to_item=rule
    # each rule is a tuple of 2 items 
    confidence=support(to_item)/support(from_item)
    if confidence>minconfidence and support(to_item)>minsupport:
        assocRules.append(rule)
assocRules



           