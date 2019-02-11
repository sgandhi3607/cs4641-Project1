import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.utils import shuffle

from sklearn import svm
import os
from textwrap import wrap


DT_DEPTH_LIMIT = 20
RANDOM_SEED = 25


columns = ['Radius','Texture','Perimeter','Area','Smoothness','Compactness',
           'Concavity','Concave_Points','Symmetry','Fractal_Dimension',
           'Malignant/Benign']

# Read CSV file into pandas df
df = pd.read_csv('../datasets/breast_cancer/breast-cancer-wisconsin.csv',
                 delimiter=',', quotechar='"', names=columns)
# Shuffle
df = shuffle(df, random_state=RANDOM_SEED)

# DROP USELESS ROWS AND COLUMNS
df.dropna(inplace=True)
cols = [0]
# Drop ID column (it's not attribute or target)
df.drop(df.columns[cols],axis=1,inplace=True)
# Drop all data points with missing variables  (denoted by '?' entry)
nostrings_row_list = [x.isdigit() for x in df.iloc[:,5]]
df = df[nostrings_row_list]


# Handle categorical data
df = pd.get_dummies(df)

print "Printing df.shape after handling categorical data: ", df.shape

# Split data into X and y vectors
X = df.ix[:, df.columns != 'Malignant/Benign']
y = df['Malignant/Benign']

# Change 2 -> 0 (benign) and 4 -> 1 (malignant)
y.replace(2, 0, inplace=True)
y.replace(4, 1, inplace=True)



X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.30, random_state=RANDOM_SEED)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#SVM_rbf
train_size = len(X_train)
offsets = range(int(0.1 * train_size), train_size, int(0.05 * train_size))
train_err = [0] * len(offsets)
test_err = [0] * len(offsets)
train_size = len(X_train)

for i, o in enumerate(offsets):
    print 'SVM: learning an SVM(kernel = rbf) with training_set_size=' + str(o)
    clf = svm.SVC(C=1.0, degree=3, gamma='auto', kernel='rbf')
    X_train_temp = X_train[:o].copy()
    y_train_temp = y_train[:o].copy()
    X_test_temp = X_test[:o].copy()
    y_test_temp = y_test[:o].copy()
    clf.fit(X_train_temp, y_train_temp)
    train_err[i] = mean_squared_error(y_train_temp,
                                     clf.predict(X_train_temp))
    test_err[i] = mean_squared_error(y_test_temp,
                                    clf.predict(X_test_temp))
    print 'train_err: ' + str(train_err[i])
    print 'test_err: ' + str(test_err[i])
    print '---'

# Plot results
print 'plotting results'
plt.figure()
title = 'WBCD SVM(kernel = rbf): Performance x Training Set Size'
plt.title('\n'.join(wrap(title,60)))
plt.plot(offsets, test_err, '-', label='test error')
plt.plot(offsets, train_err, '-', label='train error')
plt.legend()
plt.xlabel('Num Estimators')
plt.ylabel('Mean Square Error')
plt.savefig('plots/WBCD/SVM/censusIncome_rbf_PerformancexTrainingSetSize.png')
print 'plot complete'
### ---


#SVM_linear
train_size = len(X_train)
offsets = range(int(0.1 * train_size), train_size, int(0.05 * train_size))
train_err = [0] * len(offsets)
test_err = [0] * len(offsets)
train_size = len(X_train)
c_coefs = [0.01, 1, 10]

for c in c_coefs:
    for i, o in enumerate(offsets):
        print 'SVM: learning an SVM(kernel = linear) with training_set_size = ' + str(o) + ' and c = ' + str(c)
        clf = svm.SVC(C=c, degree=3, gamma='auto', kernel='linear')
        X_train_temp = X_train[:o].copy()
        y_train_temp = y_train[:o].copy()
        X_test_temp = X_test[:o].copy()
        y_test_temp = y_test[:o].copy()
        clf.fit(X_train_temp, y_train_temp)
        train_err[i] = mean_squared_error(y_train_temp,
                                         clf.predict(X_train_temp))
        test_err[i] = mean_squared_error(y_test_temp,
                                        clf.predict(X_test_temp))
        print 'train_err: ' + str(train_err[i])
        print 'test_err: ' + str(test_err[i])
        print '---'

    # Plot results
    print 'plotting results'
    plt.figure()
    title = 'WBCD SVM(kernel = linear, with c = ' + str(c) +'): Performance x Training Set Size'
    plt.title('\n'.join(wrap(title,60)))
    plt.plot(offsets, test_err, '-', label='test error')
    plt.plot(offsets, train_err, '-', label='train error')
    plt.legend()
    plt.xlabel('Num Estimators')
    plt.ylabel('Mean Square Error')
    plt.savefig('plots/WBCD/SVM/censusIncome_linear_' + str(c) + '_PerformancexTrainingSetSize.png')
    print 'plot complete'
    ### ---

#SVM_poly
train_size = len(X_train)
offsets = range(int(0.1 * train_size), train_size, int(0.05 * train_size))
train_err = [0] * len(offsets)
test_err = [0] * len(offsets)
train_size = len(X_train)
degrees = [1, 2, 3, 5]

for d in degrees:
    for i, o in enumerate(offsets):
        print 'SVM: learning an SVM(kernel = poly) with training_set_size = ' + str(o) + ' and degree = ' + str(d)
        clf = svm.SVC(C=1, degree=d, gamma='auto', kernel='poly')
        X_train_temp = X_train[:o].copy()
        y_train_temp = y_train[:o].copy()
        X_test_temp = X_test[:o].copy()
        y_test_temp = y_test[:o].copy()
        clf.fit(X_train_temp, y_train_temp)
        train_err[i] = mean_squared_error(y_train_temp,
                                         clf.predict(X_train_temp))
        test_err[i] = mean_squared_error(y_test_temp,
                                        clf.predict(X_test_temp))
        print 'train_err: ' + str(train_err[i])
        print 'test_err: ' + str(test_err[i])
        print '---'

    # Plot results
    print 'plotting results'
    plt.figure()
    title = 'WBCD SVM(kernel = poly, with degree = ' + str(d) +'): Performance x Training Set Size'
    plt.title('\n'.join(wrap(title,60)))
    plt.plot(offsets, test_err, '-', label='test error')
    plt.plot(offsets, train_err, '-', label='train error')
    plt.legend()
    plt.xlabel('Num Estimators')
    plt.ylabel('Mean Square Error')
    plt.savefig('plots/WBCD/SVM/censusIncome_poly_' + str(d) + '_PerformancexTrainingSetSize.png')
    print 'plot complete'
    ### ---
