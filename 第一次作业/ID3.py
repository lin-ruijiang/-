#ID3算法实现代码
import pandas as pd
import numpy as np
import time
import treeplotter as treeplotter

def getData(src):
    data = pd.read_csv(src,sep=' ')
    row = data.shape[1]
    testdata = pd.read_csv(src,sep=' ', names=[i for i in range(row)])

    # 获取测试数据集特征
    #feature = np.array(testdata.keys())
    #feature = np.array(feature[1:feature.size])

    # 将测试数据转换成数组
    S = np.array(testdata)
    # S = np.array(S[:, 1:S.shape[1]])

    dnaclass = S[:,-1]   #类别
    S = S[:,:-1]         #数据
    m = S.shape[0]
    n = S.shape[1]

    m_new = int(m*n/3)   #预处理后数据集一共有多少数据
    S_2 = S.reshape((m_new,3))  # S->[m_new,3]
    temp = np.array([[4],[2],[1]])  #temp是一个[3,1]的三维矩阵
    S_3 = np.dot(S_2,temp)  
    m_new = int(n/3)     #处理后的碱基，每行有60个
    S_4 = S_3.reshape((m,m_new))  #将所有的碱基转化为2000*60

    feature = [i for i in range(1,61)]
    feature = np.array(feature)
    S = np.c_[S_4,dnaclass]
    return S,feature

#统计某一特征的各个取值的概率
def Probability(x):
    value = np.unique(np.array(x))  #统计某列特征取值类型
    valueCount = np.zeros(value.shape[0]).reshape(1,value.shape[0])
    for i in range(0, value.shape[0]):
        q = np.matrix(x[np.where(x[:,0] == value[i])[0]])
        valueCount[:,i] = q.shape[0]
    p = valueCount/valueCount.sum()
    return p

#计算Entropy
#S为矩阵类型
#返回entropy
def Entropy(S):
    P = Probability(S[:,S.shape[1]-1])
    logP = np.log(P)
    entropy = -np.dot(P,np.transpose(logP))[0][0]
    return entropy

#计算EntropyA
#S为数组类型
#返回最小的信息熵的特征的索引值
#返回最小的信息熵值
def getMinEntropyA(S):
    entropy = np.zeros(S.shape[1]-1)
    for i in range(0, S.shape[1]-1):
        value = np.unique(np.array(S[:,i]))
        valueEntropy = np.zeros(value.shape[0]).reshape(1,value.shape[0])
        for j in range(0,value.shape[0]):
            q = np.matrix(S[np.where(S[:,i] == value[j])[0]])
            valueEntropy[:,j] = Entropy(q)
        proportion = Probability(np.matrix(S[:, i]).transpose())
        entropy[i] = np.dot(proportion, valueEntropy.transpose())[0][0]
    minEntropyA = entropy.min()
    positionMinEntropA = entropy.transpose().argmin()
    return positionMinEntropA, minEntropyA

#计算Gain
#S为数组类型
#返回最大信息增益的特征的索引值
def getMaxGain(S):
    entropyS = Entropy(np.matrix(S))
    positionMinEntropyA, entropyA = getMinEntropyA(S)
    if(entropyS - entropyA > 0):
        return positionMinEntropyA

#ID3算法
#返回ID3决策树
def id3Tree(S,features):
    if(Entropy(np.matrix(S)) == 0):
        return S[0][S.shape[1] - 1]
    elif features.size == 1:
        typeValues = np.unique(S[:, S.shape[1]-1])
        max = 0
        maxValue = S[0][S.shape[1] - 1]
        for value in typeValues:
            Stemp = np.array(S[np.where(S[:, S.shape[1]-1] == value)[0]])
            if max < Stemp.shape[0]:
                max = Stemp.shape[0]
                maxValue = value
        return maxValue
    else:
        bestFeatureIndex = getMaxGain(S)
        bestFeature = features[bestFeatureIndex]
        bestFeatureValues = np.unique(S[:, bestFeatureIndex])
        # 划分S，feature
        features = delArrary(features, bestFeatureIndex)
        id3tree = {bestFeature:{}}
        for value in bestFeatureValues:
            Stemp = np.array(S[np.where(S[:, bestFeatureIndex] == value)[0]])
            Stemp = delArrary(Stemp, bestFeatureIndex)
            id3tree[bestFeature][value] = id3Tree(Stemp,features)
        return id3tree

#删除数组中的某一列,维数小于2
def delArrary(arrary,index):
    if(arrary.shape[0] == arrary.size):
        x = np.array(arrary[0:index])
        y = np.array(arrary[index+1:arrary.shape[0]])
        return np.array(np.append(x,y))
    else:
        x = np.array(arrary[:,0:index])
        y = np.array(arrary[:,index+1:arrary.shape[1]])
        return np.array(np.hstack((x,y)))


'''
tree: 待测试的预训练决策树，feature：标签集，S1：单个测试样本
输入测试样本，返回测试样本的预测标签
'''
def Classify(tree, feature, S1):
    
    firstStr = list(tree.keys())[0]  # 取第一个分类特征（根结点）
    secondDict = tree[firstStr]      # 取根结点的子树集
    index = list(feature).index(firstStr)  # 第一个分类特征的索引
    classlabel = ""
    for key in secondDict.keys():
        # 在子结点中找到待分类数据的对应特征值，如果是叶节点，则直接返回其标签
        # 如果是子树，则继续递归查找，直至搜索到叶结点为止

        if S1[index] == key:
            if type(secondDict[key]).__name__ == 'dict':
                #print(secondDict[key])
                classlabel = Classify(secondDict[key],feature,S1)
            else:
                classlabel = secondDict[key]
    return classlabel

S, feature = getData('dna.data')

#生成决策树
tree = id3Tree(S, feature)
treeplotter.createPlot(tree)
print('决策树生成')
#进行预测
S, feature = getData('dna.test')
cnt = S.shape[0]
print(cnt)
right = 0
for S1 in S:
    lable = S1[-1]
    infer = Classify(tree,feature,S1[:-1])
    if lable == infer:
        right = right+1
print(right)
hit = right/cnt
print('%.2f%%' % (hit * 100)) 