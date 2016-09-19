from math import log
import operator

def createDataSet():
	dataSet = [[1, 1, 'yes'],
			   [1, 1, 'yes'],
			   [1, 0, 'no'],
			   [0, 1, 'no'],
			   [0, 1, 'no']]
	labels = ['no surfacing', 'flippers']
	return dataSet, labels

# 计算给定数据集的香农熵
def calcShannonEnt(dataSet):
	numEntries = len(dataSet)
	labelCounts = {}
	for featVec in dataSet:
		currentLabel = featVec[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1
	shannonEnt = 0.0
	for key in labelCounts:
		prob = float(labelCounts[key])/numEntries
		shannonEnt -= prob * log(prob, 2)
	return shannonEnt

# 按照给定特征划分数据集
def splitDataSet(dataSet, axis, value):
	retDataSet = []
	for featVec in dataSet:
		if featVec[axis] == value:
			reducedFeatVec = featVec[:axis]
			reducedFeatVec.extend(featVec[axis+1:])
			#reducedFeatVec和featVec的差别在于reducedFeatVec没有featVec[axis]
			retDataSet.append(reducedFeatVec)
	return retDataSet

# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    # 初始化
	numFeatures = len(dataSet[0]) - 1 
	baseEntropy = calcShannonEnt(dataSet)
	bestInfoGain = 0.0
	bestFeature = -1
    # 开始遍历
	for i in range(numFeatures):
        # i为特征值坐标，用i遍历相当于遍历每条数据的某单个特征
		featList = [example[i] for example in dataSet]
        # uniqueVals存储的是这个特征值在这个数据集中所有不重复的可能取值
		uniqueVals = set(featList)
		newEntropy = 0.0
        # 遍历当前特征中的所有唯一属性值，对每个属性值划分一次数据集，然后计算新熵值
		for value in uniqueVals:
			subDataSet = splitDataSet(dataSet, i, value)
            # 加权求和
			prob = len(subDataSet)/float(len(dataSet))
			newEntropy += prob * calcShannonEnt(subDataSet)
        # 最后所得新熵值即是由该特征下每一唯一特征值进行划分计算的香农熵的加权求和
        # 信息增益是熵的减少或者是数据无序度的减少，所以infoGain即是按该特征值划分之后较之前的信息增益
		infoGain = baseEntropy - newEntropy
		if infoGain > bestInfoGain:
			bestInfoGain = infoGain
			bestFeature = i
	return bestFeature

def majorityCnt(classList):
	classCount = {}
	for vote in classList:
		if vote not in classCount.keys(): 
			classCount[vote] = 0
		classCount[vote] += 1
	sortedClassCount = sorted(iter(classCount.items()), key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]

# 创建树的函数
# 两个参数：数据集和特征名称列表
# 特征名称列表包含数据集中所有特征的名称，算法本身不需要这个变量，提供仅是为了给出数据明确的含义
def createTree(dataSet, labels):
	# 获得数据集的所有类标签
	classList = [example[-1] for example in dataSet]
	# 如果所有类别完全相同则停止划分，直接返回该标签
	if classList.count(classList[0]) == len(classList):
		return classList[0]
	# 如果已经使用完了所有特征（数据集中只剩一列，就是末尾的类标签），则按照多数表决法返回出现次数最多的标签
	if len(dataSet[0]) == 1:
		return majorityCnt(classList)
	# 开始创建树
	# 选择最合适的划分特征(bestFeat是该特征在数据集中的列号,int)
	bestFeat = chooseBestFeatureToSplit(dataSet)
	# 根据bestFeat从特征名称列表中获得该特征的名称
	bestFeatLabel = labels[bestFeat]
	# 采用字典存储树，键值为划分的特征名
	myTree = {bestFeatLabel : {}}
	# 删除特征名称列表中的该特征
	del labels[bestFeat]
	# 获得该数据集中该特征下的所有属性值，并利用set去重
	featValues = [example[bestFeat] for example in dataSet]
	uniqueVals = set(featValues)
	# 对于每一个不同的属性
	for value in uniqueVals:
		# 复制labels
		subLabels = labels[:]
		# myTree在当前最好特征下的值是另一个以该特征的某一个属性值为键值的字典。它的值是myTree的子树（或者叶子节点，返回的是分类名称）
		myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
	return myTree

def classify(inputTree, featLabels, testVec):
	firstStr = list(inputTree.keys())[0]
	secondDict = inputTree[firstStr]
	featIndex = featLabels.index(firstStr)
	for key in secondDict.keys():
		if testVec[featIndex] == key:
			if type(secondDict[key]).__name__ == 'dict':
				return classify(secondDict[key], featLabels, testVec)
			else:
				return secondDict[key]

import json
def storeTree(inputTree, filename):
	fw = open(filename, 'w')
	json.dump(inputTree, fw)
	fw.close()

def grabTree(filename):
	fr = open(filename)
	return json.load(fr)
