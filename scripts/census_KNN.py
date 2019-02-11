import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle

import os
from textwrap import wrap


import os
from textwrap import wrap

NUMBER_OF_KFOLDS = 5
RANDOM_SEED = 25


df = pd.read_csv('../datasets/us_income/adult-train.csv', delimiter=',', quotechar='"')

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

#kNN
ks = range(1, 21)
train_err = [0] * len(ks)
test_err = [0] * len(ks)

for i, k in enumerate(ks):
    kfold_error_train = list()
    kfold_error_test = list()

    print 'kNN: learning a kNN classifier with k = ' + str(k)

    for train, test in kfold.split(df):
        clf = KNeighborsClassifier(n_neighbors = k)
        clf.fit(X_train, y_train)

        kfold_error_train.append(mean_squared_error(y_train,
                                         clf.predict(X_train)))
        kfold_error_test.append(mean_squared_error(y_test,
                                        clf.predict(X_test)))

    # Average error from all k folds,
    train_err[i] = (sum(kfold_error_train) / len(kfold_error_train))
    test_err[i] = (sum(kfold_error_test) / len(kfold_error_test))

    print 'train_err: ' + str(train_err[i])
    print 'test_err: ' + str(test_err[i])
    print '---'


# Plot results
print 'plotting results'
plt.figure()
title = 'Census Income kNN: Performance'
plt.title('\n'.join(wrap(title,60)))
plt.plot(ks, test_err, '-', label='test error')
plt.plot(ks, train_err, '-', label='train error')
plt.legend()
plt.xlabel('Num Estimators')
plt.ylabel('Mean Square Error')
plt.savefig('plots/CensusIncome/kNN/censusIncome_kNN.png')
print 'plot complete'
### ---
