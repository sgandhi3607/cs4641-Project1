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


df = pd.read_csv('../datasets/us_income/adult-train.csv', delimiter=',', quotechar='"')

import os
from textwrap import wrap

NUMBER_OF_KFOLDS = 5
RANDOM_SEED = 25

df.replace(' <=50K', 0, inplace=True)
df.replace(' >50K', 1, inplace=True)
# df.to_csv(path_or_buf='./../Data/adult.csv', index=False)
# df.dropna(0, 'any', inplace=True)
# print(df.shape)

df = pd.get_dummies(df)
# print(df.head(5))

X = df.loc[:, df.columns != 'Income']
# X = X.loc[:, X.columns != 'Fnlwgt']
# X = X.loc[:, X.columns != 'CapitalGain']
# X = X.loc[:, X.columns != 'CapitalLoss']

y = df['Income']


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.30, random_state=RANDOM_SEED)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# prepare cross validation
kfold = KFold(NUMBER_OF_KFOLDS, True, RANDOM_SEED)

#Boosting_ADABoost
train_size = len(X_train)
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

            dt = DecisionTreeClassifier(max_depth=max_depth)
            bdt = AdaBoostClassifier(base_estimator=dt, n_estimators=o)

            bdt.fit(X_train, y_train)
            kfold_error_train.append(mean_squared_error(y_train,
                                             bdt.predict(X_train)))
            kfold_error_test.append(mean_squared_error(y_test,
                                            bdt.predict(X_test)))

            AdB_predictions = bdt.predict(X_test)
            AdB_accuracy = accuracy_score(y_test, AdB_predictions)

        # Average error from all k folds,
        train_err[i] = (sum(kfold_error_train) / len(kfold_error_train))
        test_err[i] = (sum(kfold_error_test) / len(kfold_error_test))

        print 'train_err: ' + str(train_err[i])
        print 'test_err: ' + str(test_err[i])
        print '---'
        print 'AdB accuracy: ' + str(AdB_accuracy)
        print('-----------------')


    # Plot results
    print 'plotting results'
    plt.figure()
    title = 'Census Income Boosted Decision Trees(AdaBoost, Max Depth = ' + str(max_depth) + '): Performance x Num Estimators'
    plt.title('\n'.join(wrap(title,61)))
    plt.plot(max_n_estimators, test_err, '-', label='test error')
    plt.plot(max_n_estimators, train_err, '-', label='train error')
    plt.legend()
    plt.xlabel('Num Estimators')
    plt.ylabel('Mean Square Error')
    plt.savefig('plots/CensusIncome/Boosting/CensusIncome_ADABoost' + str(max_depth) + '_PerformancexNumEstimators.png')
    print 'plot complete'
    ### ---


#Boosting_GradientBoostingClassifier
max_n_estimators = range(2, 21, 1)
train_err = [0] * len(max_n_estimators)
test_err = [0] * len(max_n_estimators)
max_depths = [4, 6, 8, 10, 12]

for max_depth in max_depths:
    for i, o in enumerate(max_n_estimators):

        kfold_error_train = list()
        kfold_error_test = list()

        print 'GradientBoostingClassifier: learning a decision tree with n_estimators=' + str(o) + ' (max_depth ' + str(max_depth) + ')'

        for train, test in kfold.split(df):

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
    title = 'Census Income Boosted Decision Trees(GradientBoostingClassifier, Max Depth = ' + str(max_depth) + '): Performance x Num Estimators'
    plt.title('\n'.join(wrap(title,63)))
    test_label = 'test error - max_depth: ' + str(max_depth)
    train_label = 'train error - max_depth: ' + str(max_depth)
    plt.plot(max_n_estimators, test_err, '-', label=test_label)
    plt.plot(max_n_estimators, train_err, '-', label=train_label)
    plt.legend()
    plt.xlabel('Num Estimators')
    plt.ylabel('Mean Square Error')
    plt.savefig('plots/CensusIncome/Boosting/CensusIncome_GradientBoostingClassifier' + str(max_depth) + '_PerformancexNumEstimators.png')
    print 'plot complete'
    ### ---
