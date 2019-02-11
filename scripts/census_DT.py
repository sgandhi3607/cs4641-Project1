import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import os

NUMBER_OF_KFOLDS = 5
RANDOM_SEED = 25

df = pd.read_csv('../datasets/us_income/adult-train.csv', delimiter=',', quotechar='"')

print(df.shape)

df.replace(' <=50K', 0, inplace=True)
df.replace(' >50K', 1, inplace=True)
# df.to_csv(path_or_buf='./../Data/adult.csv', index=False)
# df.dropna(0, 'any', inplace=True)
# print(df.shape)

df = pd.get_dummies(df)
# print(df.head(5))

# Z = df.loc[:, df.columns != 'CLASS_DISTINCT']
X = df.loc[:, df.columns != 'Income']
y = df['Income']

# print(X.head(5))
# print(y.head(5))


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.30)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# prepare cross validation
kfold = KFold(NUMBER_OF_KFOLDS, True, RANDOM_SEED)

#DecisionTreeClassifier
max_depth = range(2, 22)
train_err = [0] * len(max_depth)
test_err = [0] * len(max_depth)

for i, d in enumerate(max_depth):
    kfold_error_train = list()
    kfold_error_test = list()

    print('learning a decision tree with max_depth=', str(d))

    for train, test in kfold.split(df):

        clf = DecisionTreeClassifier(max_depth=d)
        clf = clf.fit(X_train, y_train)
        kfold_error_train.append(mean_squared_error(y_train,
                                         clf.predict(X_train)))
        kfold_error_test.append(mean_squared_error(y_test,
                                        clf.predict(X_test)))

    # Average error from all k folds,
    train_err[i] = (sum(kfold_error_train) / len(kfold_error_train))
    test_err[i] = (sum(kfold_error_test) / len(kfold_error_test))

    print('train_err: ', str(train_err[i]))
    print('test_err: ', str(test_err[i]))
    print('---')

    DT_predictions = clf.predict(X_test)
    DT_accuracy = accuracy_score(y_test, DT_predictions)
    print('Census Income Tree Accuracy:', DT_accuracy)
    print('-----------------')



# Plot results
print('plotting results')
plt.figure()
plt.title('Decision Trees: Performance by Max Depth')
plt.plot(max_depth, test_err, '-', label='test error')
plt.plot(max_depth, train_err, '-', label='train error')
plt.legend()
plt.xlabel('Max Depth')
plt.ylabel('Mean Square Error')
plt.savefig('plots/CensusIncome/DecisionTree/CensusIncome_DT_PerformancexMaxDepth.png')
# plt.show()
### ---

# prepare cross validation
kfold = KFold(NUMBER_OF_KFOLDS, True, RANDOM_SEED)

### Training trees of different training set sizes (fixed max_depth=8)
train_size = len(X_train)
offsets = range(int(0.1 * train_size), train_size, int(0.05 * train_size))
max_depth_range = range(6, 20, 2)
train_err = [0] * len(offsets)
test_err = [0] * len(offsets)

for a, d in enumerate(max_depth_range):

    print('training_set_max_size:', train_size, '\n')
    for i, o in enumerate(offsets):

        kfold_error_train = list()
        kfold_error_test = list()

        for train, test in kfold.split(df):

            print('learning a decision tree with training_set_size=', str(o))
            clf = DecisionTreeClassifier(max_depth=d)
            X_train_temp = X_train[:o].copy()
            y_train_temp = y_train[:o].copy()
            X_test_temp = X_test[:o].copy()
            y_test_temp = y_test[:o].copy()

            clf = clf.fit(X_train_temp, y_train_temp)

            kfold_error_train.append(mean_squared_error(y_train_temp,
                    clf.predict(X_train_temp)))
            kfold_error_test.append(mean_squared_error(y_test_temp,
                                             clf.predict(X_test_temp)))

        # Average error from all k folds,
        train_err[i] = (sum(kfold_error_train) / len(kfold_error_train))
        test_err[i] = (sum(kfold_error_test) / len(kfold_error_test))

        print('train_err: ', str(train_err[i]))
        print('test_err: ', str(test_err[i]))
        print('---')

        DT_predictions = clf.predict(X_test)
        DT_accuracy = accuracy_score(y_test, DT_predictions)
        print('Census Income Tree Accuracy:', DT_accuracy)
        print('-----------------')

    # Plot results
    print('plotting results')
    plt.figure()
    plt.title('Decision Trees: Performance by Training Set Size for Max Depth ' + str(d))
    plt.plot(offsets, test_err, '-', label='test error')
    plt.plot(offsets, train_err, '-', label='train error')
    plt.legend()
    plt.xlabel('Training Set Size')
    plt.ylabel('Mean Square Error')
    filename = 'CensusIncome_DT_PerformancexTrainingSetSize_MAXDEPTH=' + str(d) + '.png'
    plt.savefig('plots/CensusIncome/DecisionTree/%s' % filename)
    print('plot complete')
### ---
