import pandas as pd
import numpy as np
import csv


def underSampling(T, dataset, minorClassname, numattrs, N):
    majorityClassLabel = []
    minorityClassData = []
    underSampledMajorityData = []
    N = int(T/(N / 100))
    for index, eachSample in enumerate(dataset, 1):
        if eachSample[numattrs-1] != minorClassname:
            majorityClassLabel.append(index)
        else:
            minorityClassData.append(eachSample)
    randomMajority = np.random.choice(majorityClassLabel, N, replace=True)
    #print(type(randomMajority))
    for index, eachRow in enumerate(dataset, 1):
        for i in np.nditer(randomMajority):
            if index == i:
                underSampledMajorityData.append(eachRow)
    for eachRow in minorityClassData:
        underSampledMajorityData.append(eachRow)
    with open(".\\Syntheitc_Data.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(underSampledMajorityData)
        writer.writerows(minorityClassData)
    return underSampledMajorityData
        #Classify using C4.5 or Ripper

