import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import csv

import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import tree, linear_model

from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from scipy import interp
from random import shuffle

import numpy as np
# noinspection PyUnresolvedReferences
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


# def treeClassifier2(majoritySamples, minoritySamples):
def treeClassifier2(allSamples,numattr, sampling):
    #print(len(allSamples))
    shuffle(allSamples)
    print((allSamples[0][0]))

    nArray = np.matrix(allSamples, dtype=float)
    #nPArray = np.matrix(allSamples)
    #print (nPArray.shape)
    #labels = allSamples[:,numattr-1]
    X = (nArray[:,:numattr-1])#nPArray[:,:numattr-1]
    y = np.array(nArray[:,numattr-1])#nPArray[:,numattr-1]
    print(y)
    #labels = np.array(labels)
    #print(data)
    #print(labels)

    #X = np.matrix(data)
    #y = np.array(labels)#, dtype=int)

    kf = KFold(n_splits=10)
    clf = linear_model.LogisticRegression()

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train, test in kf.split(X, y):
        predicted = clf.fit(X[train], y[train]).predict_proba(X[test])
        #print(predicted)
        # Compute ROC curve and area the curve
        #print("AUC[" + str(i) + "]: " + str(metrics.roc_auc_score(y[test], predicted[:, 1])))
        #print ("Accuracy: " + str(metrics.accuracy_score(y[test], predicted[:, 1].round())))

        fpr, tpr, thresholds = metrics.roc_curve(y[test], predicted[:, 1], pos_label=1)

        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        #plt.plot(fpr, tpr, lw=1, alpha=0.3,
        #         label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        #plt.plot(fpr * 100, tpr * 100)

        i += 1

        print("FAccuracy: " + str(metrics.accuracy_score(y[test], predicted[:, 1].round())))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    print('AUC SCORE of ', sampling, ' is ' , mean_auc)
    std_auc = np.std(aucs)
    with open(".\\ROCValue.txt", 'wb') as f:
        np.savetxt(f, np.r_[mean_fpr,mean_tpr])
    open('.\\Syntheitc_Data.txt', 'a').writelines([l for l in open('.\\ROCValue.txt').readlines()])

def plotConvexHull(fprs, tprs):
    # points = np.random.rand(4, 2)  # 30 random points in 2-D
    points = np.column_stack((fprs, tprs))
    # print(points)

    hull = ConvexHull(points)

    plt.plot(points[:, 0], points[:, 1], 'o')

    # print (points[:, 0])
    # print (points[:, 1])

    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

    plt.title('Convex Hull')
    plt.show()

def plotData(filename):
    meanFPRFirst = []
    meanTPRFirst = []
    meanFPRSecond = []
    meanTPRSecond = []
    meanFPRNaive = []
    meanTPRNaive = []
    with open(filename) as f:
        for linenumber, line in enumerate(f):
            if linenumber < 100:
                meanFPRFirst.append(line)
            elif linenumber >= 100 and linenumber < 200:
                meanTPRFirst.append(line)
            elif linenumber >= 200 and linenumber < 300:
                meanFPRSecond.append(line)
            elif linenumber >= 300 and linenumber < 400:
                meanTPRSecond.append(line)
            elif linenumber >= 400 and linenumber < 500:
                meanFPRNaive.append(line)
            elif linenumber >= 500 and linenumber < 600:
                meanTPRNaive.append(line)

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Random Guess', alpha=.8)
    plt.plot(meanFPRFirst, meanTPRFirst, color='b',
             label='Undersampling',
             lw=2, alpha=.8)
    plt.plot(meanFPRNaive, meanTPRNaive, color='k',
             label="Naive-Bayes",
             lw=2, alpha=.8)
    plt.plot(meanFPRSecond, meanTPRSecond, color='r',
             label='Undersampling and SMOTE',
             lw=2, alpha=.8)
    plt.rcParams['font.size'] = 12
    plt.title('ROC curve for Forest Cover')
    plt.xlabel('% False Positive')
    plt.ylabel('% True Positive')
    plt.grid(True)
    plt.legend(loc=0)
    plt.show()
    #plotConvexHull(meanFPRFirst, meanTPRFirst)
    #plotConvexHull(meanFPRSecond, meanTPRSecond)
