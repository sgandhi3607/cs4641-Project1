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

NUMBER_OF_KFOLDS = 5
RANDOM_SEED = 25

from sklearn.model_selection import cross_validate

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


X = df.loc[:, df.columns != 'Malignant/Benign']
y = df['Malignant/Benign']


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.30, random_state=30)

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
title = 'WBCD kNN: Performance'
plt.title('\n'.join(wrap(title,60)))
plt.plot(ks, test_err, '-', label='test error')
plt.plot(ks, train_err, '-', label='train error')
plt.legend()
plt.xlabel('Num Estimators')
plt.ylabel('Mean Square Error')
plt.savefig('plots/WBCD/kNN/WBCD_kNN.png')
print 'plot complete'
### ---
