import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from textwrap import wrap

df = pd.read_csv('../datasets/us_income/adult-train.csv', delimiter=',', quotechar='"')

import os
from textwrap import wrap

RANDOM_SEED = 25


df.replace(' <=50K', 0, inplace=True)
df.replace(' >50K', 1, inplace=True)

df = pd.get_dummies(df)
# print(df.head(5))

X = df.loc[:, df.columns != 'Income']
y = df['Income']

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
# activation_functions = ['relu']

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
           hidden_layer_sizes=(13, 13, 13), learning_rate='constant',
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

        NN_predictions = mlp.predict(X_test_temp)
        NN_accuracy = accuracy_score(y_test_temp, NN_predictions)

        # print(classification_report(y_train, mlp.predict(X_train)))
        # print(classification_report(y_test, mlp.predict(X_test)))

        print 'train_err: ' + str(train_err[i])
        print 'test_err: ' + str(test_err[i])
        print '---'
        print 'NN accuracy: ' + str(NN_accuracy)
        print('-----------------')

    # Plot results
    print 'plotting results'
    plt.figure()
    title = 'Census Income (Fewer hidden neurons): Performance x Training Set Size using Activation ' + activation
    plt.title('\n'.join(wrap(title,60)))
    # plt.subplots_adjust(top=0.85)
    plt.plot(offsets, test_err, '-', label='test error')
    plt.plot(offsets, train_err, '-', label='train error')
    plt.legend()
    plt.xlabel('Training Set Size')
    plt.ylabel('Mean Square Error')
    filename = 'censusIncome' + activation + '_newArchitecture_PerformancexTrainingSetSize.png'
    plt.savefig('plots/CensusIncome/NN/old_architecture_zeroes_and_one/' + filename)
    print 'plot complete'
    ### ---
