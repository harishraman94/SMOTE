from Smote import SMOTE
from Classifier import NaiveBayes
from Classifier import C45
#from OverSampling import OverSampling
from RandomUnderSampling import UnderSampling
import csv

def main():
    dataFile1='.\\ForestCover.csv'
    classColumnNumber = 55
    minorClassValue='1'
    numattr=55

    dataFile2 = '.\\Mammography.csv'
    classColumnNumberMammo = 7
    minorClassValueMammo = '1'
    numattrMammo = 7

    dataFile3 = '.\\phoneme.csv'
    classColumnNumberPhoneme = 6
    minorClassValuePhoneme = '1'
    numattrPhoneme = 6

    performSampling(dataFile1,classColumnNumber, numattr,minorClassValue)
    C45.plotData('.\\Syntheitc_Data.txt')


# appends the original dataset with the newly generated value to give final dataset used for classification
def generateFinalSyntheticDataset(dataSet, synthetic_dataset):
    for eachRow in synthetic_dataset:
        dataSet.append(eachRow)
    return dataSet


def createNewDatasetFileUnderSampling(dataSet):
    newDataset = '.\\Undersampling.csv'
    with open(newDataset, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(dataSet)

def createNewDatasetFileUnderSamplingSmote(dataset):
    newDataset = '.\\UndersamplingSmote.csv'
    with open(newDataset, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(dataset)

def getSeparatedSamples(filename,label_column,minorityClassName):
    minoritySamples = []
    majoritySamples = []
    dataset=[]
    with open(filename, 'r') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            #Sample = row[0].split(',')
            if (row[label_column-1] == minorityClassName):
                # minorityCounter += 1
                minoritySamples.append(row)
                # print (Sample[0:8])

                # print(row[0][2])
                # print (', '.join(row))
            else:
                # majorityCounter += 1
                majoritySamples.append(row)
            dataset.append(row)
    csvfile.close
    return dataset,minoritySamples, majoritySamples


def performSampling(dataFile,classColumnNumber,numattrs,minorClassValue,N=400):
    dataSet, MinorityData, MajorityData = getSeparatedSamples(dataFile, classColumnNumber, minorClassValue)
    NumberofMinorSamples = len(MinorityData)

    print("Number of Minor samples present in the Dataset: ", NumberofMinorSamples)

    #Only UnderSampling
    uSamplingDataset = UnderSampling.underSampling(len(MinorityData), dataSet, minorClassValue, numattrs, N)
    createNewDatasetFileUnderSampling(uSamplingDataset)
    C45.treeClassifier2(dataSet, numattrs, 'UnderSampling')

    #UnderSampling and SMOTE
    underSamplingDataset = UnderSampling.underSampling(len(MinorityData), dataSet, minorClassValue, numattrs, N)
    uSDataset = SMOTE.SMOTE(len(MinorityData), N, MinorityData, numattrs, underSamplingDataset, 5, minorClassValue)
    uSmoteDataset = generateFinalSyntheticDataset(underSamplingDataset, uSDataset)
    createNewDatasetFileUnderSamplingSmote(uSmoteDataset)
    C45.treeClassifier2(uSmoteDataset, numattrs, 'SMOTEUndersampling')

    #NaiveBayes
    NaiveBayes.naiveBayes(MajorityData, MinorityData, numattrs)
if __name__ == "__main__":
    main()
