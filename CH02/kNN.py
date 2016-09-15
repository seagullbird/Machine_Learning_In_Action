# -*- coding: utf-8 -*-
from numpy import *
import operator
from os import listdir

def createDataSet():
	group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
	labels = ['A', 'A', 'B', 'B']
	return group, labels

def classify0(inX, dataSet, labels, k):
	# 距离计算
	
	# dataSet参数传入的是一个array类型，其shape属性表示数组的维度。这是一个指示数组在每个维度上大小的整数元组。shape[0]为行数，shape[1]为列数
	dataSetSize = dataSet.shape[0]

	# tile(A, reps): 重复A, reps次。reps可以是一个int也可以是一个tuple。
	# 这句把传入的待分类向量纵方向上重复了dataSetSize次，使其成为一个和训练集dataSet相同大小的矩阵，便于之后的计算
	# 从减去dataSet这一步开始，都在进行欧拉距离公式的计算
	diffMat = tile(inX, (dataSetSize, 1)) - dataSet
	sqDiffSet = diffMat**2	# **:求幂运算，前底数后指数
	sqDistances = sqDiffSet.sum(axis=1)
	distances = sqDistances**0.5
	# 上面两句返回纵方向（axis=1）的和并开平方，完成距离计算并返回在一个数组

	# 这句将得到的距离数组排序（从小到大），返回的是排序后的元素在原数组中的序号组成的数组（这些序号同时也是训练集对应的labels在labels数组中的序号）
	sortedDistIndicies = distances.argsort()

	# 选择距离最小的k个点
	# 计算距离前k小元素中各标签label的出现次数，用标签－次数键值对存储在classCount中
	classCount = {}
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
	# 排序
	# 将classCount按值的大小从大到小排序
	# key参数说明按待排序对象的第二个属性进行排序
	sortedClassCount = sorted(iter(classCount.items()), key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]

def file2matrix(filename):
	fr = open(filename)
	# readlines()将文件的每一行（包括'\n'）作为一个str存在一个list中并返回这个list
	arrayOLines = fr.readlines()
	# 所以len可以得到文件的行数
	numberOfLines = len(arrayOLines)
	# zeros根据传入的tuple作为大小返回一个数值全0的矩阵，这里返回的是3列，numberOfLines行的矩阵
	returnMat = zeros((numberOfLines, 3))
	classLabelVector = []
	index = 0
	for line in arrayOLines:
		# strip()函数去掉后面的'\n'
		line = line.strip()
		# 按'\t'分开这个字符串并将分开后的元素保存到列表
		listFromLine = line.split('\t')
		returnMat[index, :] = listFromLine[0:3]
		classLabelVector.append(int(listFromLine[-1]))
		index += 1
	return returnMat, classLabelVector

def autoNorm(dataSet):
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	normDataSet = zeros(shape(dataSet))
	m = dataSet.shape[0]
	normDataSet = dataSet - tile(minVals, (m, 1))
	normDataSet = normDataSet / tile(ranges, (m, 1))
	return normDataSet, ranges, minVals

def datingClassTest():
	# 测试样本集占总样本集的比例
	hoRatio = 0.10
	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	m = normMat.shape[0]
	numTestVecs = int(m * hoRatio)
	errorCount = 0.0
	for i in range(numTestVecs):
		classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 5)
		print("The classifer came back with : %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
		if classifierResult != datingLabels[i]:
			errorCount += 1.0
	print("The total error rate is: %f"  %  (errorCount/float(numTestVecs)))

def classifyPerson():
	resultList = ['not at all', 'in small doses', 'in large doses']
	percentTats = float(input('Percentage of time spent playing video games?'))
	ffMiles = float(input('Frequent flier miles earned per year?'))
	iceCream = float(input('Liters of ice cream consumed per year?'))
	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	inArr = array([ffMiles, percentTats, iceCream])
	classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
	print('You will probably like this person: ', resultList[classifierResult-1])

def img2vector(filename):
	returnVect = zeros((1, 1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVect[0, 32*i+j] = int(lineStr[j])
	return returnVect

def handwritingClassTest():
	hwLabels = []
	trainingFileList = listdir('trainingDigits')
	m = len(trainingFileList)
	trainingMat = zeros((m, 1024))
	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		hwLabels.append(classNumStr)
		trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
	testFileList = listdir('testDigits')
	errorCount = 0.0
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
		classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
		print('The classifer came back with: %d, the real answer is: %d' % (classifierResult, classNumStr))
		if classifierResult != classNumStr:
			errorCount += 1.0
	print('\nThe total number of errors is: %d' % errorCount)
	print('\nThe total error rate is: %f' % (errorCount/float(mTest)))