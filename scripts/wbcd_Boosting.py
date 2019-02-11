import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import shuffle


import os
from textwrap import wrap

NUMBER_OF_KFOLDS = 5
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


# df = pd.get_dummies(df)
# print(df.head(5))

X = df.loc[:, df.columns != 'Malignant/Benign']
y = df['Malignant/Benign']


# prepare cross validation
kfold = KFold(NUMBER_OF_KFOLDS, True, RANDOM_SEED)

#Boosting_ADABoost
# train_size = len(X_train)
max_n_estimators = range(2, 31, 1)
train_err = [0] * len(max_n_estimators)
test_err = [0] * len(max_n_estimators)

max_depths = [4, 6, 8]

for max_depth in max_depths:
    for i, o in enumerate(max_n_estimators):

        kfold_error_train = list()
        kfold_error_test = list()

        print 'AdaBoostClassifier: learning a decision tree with n_estimators=' + str(o) + ' (max_depth ' + str(max_depth) + ')'

        for train, test in kfold.split(df):

            scaler = StandardScaler()

            X_train = X.iloc[train]
            X_test = X.iloc[test]

            y_train = y.iloc[train]
            y_test = y.iloc[test]

            scaler.fit(X_train)

            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)


            dt = DecisionTreeClassifier(max_depth=max_depth)
            bdt = AdaBoostClassifier(base_estimator=dt, n_estimators=o)

            bdt.fit(X_train, y_train)
            kfold_error_train.append(mean_squared_error(y_train,
                                             bdt.predict(X_train)))
            kfold_error_test.append(mean_squared_error(y_test,
                                            bdt.predict(X_test)))


        # Average error from all k folds,
        train_err[i] = (sum(kfold_error_train) / len(kfold_error_train))
        test_err[i] = (sum(kfold_error_test) / len(kfold_error_test))

        # print "train error list: ", kfold_error_train
        # print "test error list: ", kfold_error_test

        print 'train_err: ' + str(train_err[i])
        print 'test_err: ' + str(test_err[i])
        print '---'

    # Plot results
    print 'plotting results'
    plt.figure()
    title = 'WBCD Boosted Decision Trees(AdaBoost, Max Depth = ' + str(max_depth) + '): Performance x Num Estimators'
    plt.title('\n'.join(wrap(title,61)))
    plt.plot(max_n_estimators, test_err, '-', label='test error')
    plt.plot(max_n_estimators, train_err, '-', label='train error')
    plt.legend()
    plt.xlabel('Num Estimators')
    plt.ylabel('Mean Square Error')
    plt.savefig('plots/WBCD/Boosting/WBCD_ADABoost' + str(max_depth) + '_PerformancexNumEstimators.png')
    print 'plot complete'
    ### ---


#Boosting_GradientBoostingClassifier
max_n_estimators = range(2, 21, 1)
train_err = [0] * len(max_n_estimators)
test_err = [0] * len(max_n_estimators)
max_depths = [4, 6, 8]

for max_depth in max_depths:
    for i, o in enumerate(max_n_estimators):
        kfold_error_train = list()
        kfold_error_test = list()
        print 'GradientBoostingClassifier: learning a decision tree with n_estimators=' + str(o) + ' (max_depth ' + str(max_depth) + ')'
        for train, test in kfold.split(df):

            scaler = StandardScaler()

            X_train = X.iloc[train]
            X_test = X.iloc[test]

            y_train = y.iloc[train]
            y_test = y.iloc[test]

            scaler.fit(X_train)

            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            bdt = GradientBoostingClassifier(max_depth=max_depth, n_estimators=o)
            bdt.fit(X_train, y_train)

        kfold_error_train.append(mean_squared_error(y_train,
                                         bdt.predict(X_train)))
        kfold_error_test.append(mean_squared_error(y_test,
                                        bdt.predict(X_test)))

        # Average error from all k folds,
        train_err[i] = (sum(kfold_error_train) / len(kfold_error_train))
        test_err[i] = (sum(kfold_error_test) / len(kfold_error_test))

        print 'train_err: ' + str(train_err[i])
        print 'test_err: ' + str(test_err[i])
        print '---'

    # Plot results
    print 'plotting results'
    plt.figure()
    title = 'WBCD Boosted Decision Trees(GradientBoostingClassifier, Max Depth = ' + str(max_depth) + '): Performance x Num Estimators'
    plt.title('\n'.join(wrap(title,63)))
    plt.plot(max_n_estimators, test_err, '-', label='test error')
    plt.plot(max_n_estimators, train_err, '-', label='train error')
    plt.legend()
    plt.xlabel('Num Estimators')
    plt.ylabel('Mean Square Error')
    plt.savefig('plots/WBCD/Boosting/WBCD_GradientBoostingClassifier' + str(max_depth) + '_PerformancexNumEstimators.png')
    print 'plot complete'
    ### ---
