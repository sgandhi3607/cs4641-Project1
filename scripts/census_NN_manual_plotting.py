import matplotlib.pyplot as plt
from textwrap import wrap

relu_layer1_size = [x for x in range(10, 90, 10)]
print relu_layer1_size

relu_train_error = [0.12211064797271694, 0.10941644562334217, 0.09842743463433119, 0.08687002652519894,
                    0.08772262220538082, 0.07351269420234938, 0.006688139446760137, 0.06205001894657067]

relu_test_error = [0.16145430434302133, 0.16918996574207096, 0.17637307989833131, 0.18267211846612885, 0.17515747596419493,
                    0.1889711570339264, 0.18311415626036026, 0.1846612885401702 ]

print 'plotting results'
plt.figure()
title = 'Census Income NN RELU: Performance x Num of  Nodes in Hidden Layer 1 '
plt.title('\n'.join(wrap(title,63)))
plt.plot(relu_layer1_size, relu_test_error, '-', label='test error')
plt.plot(relu_layer1_size, relu_train_error, '-', label='train errr')
plt.legend()
plt.xlabel('number of nodes in hidden layer 1')
plt.ylabel('Mean Square Error')
plt.savefig('plots/CensusIncome/NN/zeroes_and_one/varyingArchitecture/Manual' + '_RELU_VaryHiddenLayer.png')
print 'plot complete'

#
# logistic_train_error = [0.12211064797271694, 0.10941644562334217, 0.09842743463433119, 0.08687002652519894,
#                     0.08772262220538082, 0.07351269420234938, 0.006688139446760137, 0.06205001894657067]
#
# logistic_test_error = [0.12211064797271694, 0.10941644562334217, 0.09842743463433119, 0.08687002652519894,
#                     0.08772262220538082, 0.07351269420234938, 0.006688139446760137, 0.06205001894657067]
#
#
#
#
# print 'plotting results'
# plt.figure()
# title = 'Census Income NN Logistic: Performance x Num of  Nodes in Hidden Layer 1 '
# plt.title('\n'.join(wrap(title,63)))
# plt.plot(relu_layer1_size, logistic_train_error, '-', label='train error')
# plt.plot(relu_layer1_size, logistic_test_error, '-', label='test error')
# plt.legend()
# plt.xlabel('number of nodes in hidden layer 1')
# plt.ylabel('Mean Square Error')
# plt.savefig('plots/CensusIncome/NN/zeroes_and_one/varyingArchitecture/Manual' + '_Logistic_VaryHiddenLayer.png')
# print 'plot complete'
#
# tanh_test_error = [0.12211064797271694, 0.10941644562334217, 0.09842743463433119, 0.08687002652519894,
#                     0.08772262220538082, 0.07351269420234938, 0.006688139446760137, 0.06205001894657067]
#
# tanh_test_error = [0.12211064797271694, 0.10941644562334217, 0.09842743463433119, 0.08687002652519894,
#                     0.08772262220538082, 0.07351269420234938, 0.006688139446760137, 0.06205001894657067]
#
#
# print 'plotting results'
# plt.figure()
# title = 'Census Income NN Tanh: Performance x Num of  Nodes in Hidden Layer 1 '
# plt.title('\n'.join(wrap(title,63)))
# plt.plot(relu_layer1_size, tanh_train_error, '-', label='train error')
# plt.plot(relu_layer1_size, tanh_test_error, '-', label='test error')
# plt.legend()
# plt.xlabel('number of nodes in hidden layer 1')
# plt.ylabel('Mean Square Error')
# plt.savefig('plots/CensusIncome/NN/zeroes_and_one/varyingArchitecture/Manual' + '_Tanh_VaryHiddenLayer.png')
# print 'plot complete'
#
# #
