import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# from sklearn.cross_validation import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

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

# Split into 30%  training data, 70% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.30, random_state=RANDOM_SEED)


# Apply scaling. Large values of certain features undesireable for NN
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# NNClassifier
train_size = len(X_train)
offsets = range(int(0.1 * train_size), int(train_size), int(0.1 * train_size))


# mlp = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
#        beta_2=0.999, epsilon=1e-08,
#        hidden_layer_sizes=(13, 13, 13), learning_rate='constant',
#        learning_rate_init=0.001, max_iter=500, momentum=0.9,
#        nesterovs_momentum=True, random_state=None,
#        shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1)
# mlp.fit(X_train,y_train)

# predictions = mlp.predict(X_test)
# print(classification_report(y_test,predictions))


train_err = [0] * len(offsets)
test_err = [0] * len(offsets)

print 'training_set_max_size:', train_size, '\n'

activation_functions = ['relu', 'logistic', 'tanh']

for activation in activation_functions:
    for i, o in enumerate(offsets):
        print 'activation: ' + activation
        print 'learning a neural net with training_set_size=' + str(o)
        print 'getting data',
        X_train_temp = X_train[:o].copy()
        y_train_temp = y_train[:o].copy()
        X_test_temp = X_test[:o].copy()
        y_test_temp = y_test[:o].copy()
        print 'building net',
        mlp = MLPClassifier(activation=activation, alpha=0.0001, batch_size='auto', beta_1=0.9,
           beta_2=0.999, epsilon=1e-08,
           # hidden_layer_sizes=(50, 25, 12),
           hidden_layer_sizes=(13, 13, 13),
           learning_rate='constant',
           learning_rate_init=0.001, max_iter=500, momentum=0.9,
           nesterovs_momentum=True, random_state=None,
           shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1)
        print 'training',
        mlp.fit(X_train,y_train)
        print 'validating'
        train_err[i] = mean_squared_error(y_train_temp,
                    mlp.predict(X_train_temp))
        test_err[i] = mean_squared_error(y_test_temp,
                    mlp.predict(X_test_temp))

        # print(classification_report(y_train, mlp.predict(X_train)))
        # print(classification_report(y_test, mlp.predict(X_test)))

        print 'train_err: ' + str(train_err[i])
        print 'test_err: ' + str(test_err[i])
        print '---'

    # Plot results
    print 'plotting results'
    plt.figure()
    title = 'WBCD Neural Nets: Performance x Training Set Size using Activation ' + activation
    plt.title('\n'.join(wrap(title,60)))
    # plt.subplots_adjust(top=0.85)
    plt.plot(offsets, test_err, '-', label='test error')
    plt.plot(offsets, train_err, '-', label='train error')
    plt.legend()
    plt.xlabel('Training Set Size')
    plt.ylabel('Mean Square Error')
    filename = 'wbcd' + activation + '_PerformancexTrainingSetSize.png'
    plt.savefig('plots/WBCD/NN/' + filename)
    print 'plot complete'
    ### ---
