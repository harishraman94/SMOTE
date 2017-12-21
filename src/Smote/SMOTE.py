import csv
import random
from random import randint
import math
import operator
from sklearn.neighbors import NearestNeighbors
import csv
global dataset


def __loadDataset__(filename, number_attributes):
    csvfile = open(filename, 'r')
    lines = csv.reader(csvfile)
    dataset = list(lines)

    for x in range(len(dataset)):
        for y in range(number_attributes):
            dataset[x][y] = float(dataset[x][y])
    return dataset


def euclideanDistance(a, b, length):
    distance = 0
    for x in range(length):
        distance += pow((float(a[x]) - float(b[x])), 2)
    return math.sqrt(distance)


def getNeighbours(trainingSet, eachMinorsample, k):
    distances = []
    length = len(eachMinorsample) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(eachMinorsample, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
        distances.sort(key=operator.itemgetter(1))
        neighbours = []
    for x in range(k):
        neighbours.append(distances[x + 1][0])
    return neighbours


def seperateMinority(dataSet, MinorClassName, classColumnNumber):
    minorSamples = []
    for eachSample in dataSet:
        if (eachSample[classColumnNumber] == MinorClassName):
            minorSamples.append(eachSample)
    return minorSamples


def SMOTE(T, N, minorSamples, numattrs, dataSet, k,minorClassName):
    if (N < 100):
        print("Number of sample to be generated should be more than 100%")
        raise ValueError
    N = int(N / 100) * T
    knnArrayy = []
    for eachMinor in minorSamples:
        knnArrayy = (getNeighbours(dataSet, eachMinor, k))
    #knnArrayy = NearestNeighbors(n_neighbors=k).fit(dataSet)
    return populate(N, minorSamples, knnArrayy, numattrs,minorClassName)


def populate(N, minorSample, nnarray, numattrs,MinorClassName):
    synthetic_dataset=[]
    while (N > 0):
        nn = randint(0, len(nnarray) - 2)
        eachUnit = []
        for attr in range(0, numattrs - 1):
            #print( type(minorSample[nn][attr]))
            diff = float(nnarray[nn][attr]) - float(minorSample[nn][attr])
            gap = random.uniform(0, 1)
            eachUnit.append(float(minorSample[nn][attr]) + gap * diff)
        eachUnit.append(MinorClassName)
        synthetic_dataset.append(eachUnit)
        N = N - 1
    #print (len(synthetic_dataset))
    with open(".\\Syntheitc_Data.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(synthetic_dataset)
    return synthetic_dataset
