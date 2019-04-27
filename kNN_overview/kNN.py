import numpy as np 
import operator #运算符模块

def creatDataSet():
    group = np.array([[1.0, 1.1],[1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels
#kNN分类算法
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    #距离计算
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet #tile()重复数组
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    #按距离递增次序排序
    sortedDistIndicies = distances.argsort()
    #选择距离最小的k个点
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    #排序
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
#将文本记录转化为numpy的解析程序
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        print('line:', line)
        #listFromLine = line.split('\t')
        listFromLine = line
        print('listFromline:', listFromLine)
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(listFromLine[-2])
        index += 1
    return returnMat, classLabelVector


if __name__ == "__main__":
    group, labels = creatDataSet()
    print(classify0([0,0], group, labels, 3))
    datingDataMat, datingLabels = file2matrix('datingtestset.txt')
    print(datingDataMat,'\n',datingLabels)
    print(classify0([1,2,3], datingDataMat, datingLabels, 3))