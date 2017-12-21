import numpy as np
from sklearn.naive_bayes import GaussianNB
from random import shuffle
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn import metrics
import matplotlib.pyplot as plt

def naiveBayes(majoritySamples, minoritySamples,numattr):
    allSamples = minoritySamples + majoritySamples

    shuffle(allSamples)

    nPArray = np.array(allSamples)
    data = nPArray[:, :numattr-1]
    labels = nPArray[:, numattr-1]
    #print(labels)

    X = np.array(data, dtype=float)
    y = np.array(labels, dtype=int)

    kf = KFold(n_splits=10)
    clf = GaussianNB()


    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    sample_weight = np.array([50 if i == 1 else 1 for i in y])
    #print(len(sample_weight))
    i = 0
    for train, test in kf.split(X, y):
        #print(len(X[test]))
        clf.fit(X[train], y[train], [5 if i ==1 else 1 for i in y[train]])
        predicted = clf.predict_proba(X[test])

        # Compute ROC curve and area the curve
        #print("AUC[" + str(i) + "]: " + str(metrics.roc_auc_score(y[test], predicted[:,1].round())))
        #print ("Accuracy: " + str(metrics.accuracy_score(y[test], predicted)))

        fpr, tpr, thresholds = roc_curve(y[test], predicted[:, 1].round())

        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        #plt.plot(fpr, tpr, lw=1, alpha=0.3,
        #         label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        # plt.plot(fpr * 100, tpr * 100)

        i += 1
    print("Accuracy: " + str(metrics.accuracy_score(y[test], predicted[:, 1].round())))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    print('AUC Score of NaiveBayes : ' , mean_auc)
    std_auc = np.std(aucs)
    with open(".\\ROCValue.txt", 'wb') as f:
        np.savetxt(f, np.r_[mean_fpr,mean_tpr])
    open('.\\Syntheitc_Data.txt', 'a').writelines([l for l in open('.\\ROCValue.txt').readlines()])