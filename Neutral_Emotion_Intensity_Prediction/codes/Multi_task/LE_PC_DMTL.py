
# coding: utf-8

# ## LE_PC_DMTL - A deep multi-task learning neural network that uses parallely connected components - a CNN/LSTM, fully connected layers, and a feature vector derived from linguistic lexicons and pre-trained activations of a network trained for Emoji detection (deepMoji). So, a *L*exicon and *E*moji detection based, *P*arallely *C*onnected *D*eep *M*ulti-*T*ask *L*earning neural network.
# 
# ### This architecture handles all the four (can be extended to n) emotions together in a single architecture (as opposed to the architecture in ../LE_PC_DNN/LE_PC_DNN_complete.ipynb). Details can be found in the paper.

# In[ ]:

import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout
from keras.layers import LSTM,Bidirectional,GRU,SimpleRNN
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D,MaxPooling1D, AveragePooling1D
from keras.layers import Input, merge, Dropout
from keras.models import Model
import tensorflow as tf
#tf.python.control_flow_ops = tf
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import timeit


# ## Load the pre-trained word2vec based train, dev, and test set tweets representations
# ### Please run the corresponding code in ../Supporting_Codes/ to produce these vector representations.
# #### Note that these will be vectors of the form (n, l, d) where n is the number of tweets in the set, l is the chosen maximum length (zero padded to have same sequence length = 50 for all samples), and d is the dimensionality of the word embedding (400, since we are using the Twitter word2vec model)

# In[ ]:

emotion = 'anger'
x1_train_anger = np.load('../../intermediate_files/word2vec_based_concatenated_vectors/train/'
                   +emotion+'.npy')
x1_dev_anger = np.load('../../intermediate_files/word2vec_based_concatenated_vectors/dev/'
                   +emotion+'.npy')

'''
we combine the train, dev to serve as the training vector. We have already determined the
best performing hyperparamter on the dev set, and just need to see results on test set now.
'''

x1_train_anger = np.concatenate((x1_train_anger,x1_dev_anger),axis=0)

x1_test_anger = np.load('../../intermediate_files/word2vec_based_concatenated_vectors/test/'
                   +emotion+'.npy')

print('x1_train_anger shape:', x1_train_anger.shape)    # (n, 50, 400)
print('x1_test_anger shape:', x1_test_anger.shape)      # (n, 50, 400)


# In[ ]:

emotion = 'fear'
x1_train_fear = np.load('../../intermediate_files/word2vec_based_concatenated_vectors/train/'
                   +emotion+'.npy')
x1_dev_fear = np.load('../../intermediate_files/word2vec_based_concatenated_vectors/dev/'
                   +emotion+'.npy')

'''
we combine the train, dev to serve as the training vector. We have already determined the
best performing hyperparamter on the dev set, and just need to see results on test set now.
'''

x1_train_fear = np.concatenate((x1_train_fear,x1_dev_fear),axis=0)

x1_test_fear = np.load('../../intermediate_files/word2vec_based_concatenated_vectors/test/'
                   +emotion+'.npy')

print('x1_train_fear shape:', x1_train_fear.shape)    # (n, 50, 400)
print('x1_test_fear shape:', x1_test_fear.shape)      # (n, 50, 400)


# In[ ]:

emotion = 'joy'
x1_train_joy = np.load('../../intermediate_files/word2vec_based_concatenated_vectors/train/'
                   +emotion+'.npy')
x1_dev_joy = np.load('../../intermediate_files/word2vec_based_concatenated_vectors/dev/'
                   +emotion+'.npy')

'''
we combine the train, dev to serve as the training vector. We have already determined the
best performing hyperparamter on the dev set, and just need to see results on test set now.
'''

x1_train_joy = np.concatenate((x1_train_joy,x1_dev_joy),axis=0)

x1_test_joy = np.load('../../intermediate_files/word2vec_based_concatenated_vectors/test/'
                   +emotion+'.npy')

print('x1_train_joy shape:', x1_train_joy.shape)    # (n, 50, 400)
print('x1_test_joy shape:', x1_test_joy.shape)      # (n, 50, 400)


# In[ ]:

emotion = 'sadness'
x1_train_sadness = np.load('../../intermediate_files/word2vec_based_concatenated_vectors/train/'
                   +emotion+'.npy')
x1_dev_sadness = np.load('../../intermediate_files/word2vec_based_concatenated_vectors/dev/'
                   +emotion+'.npy')

'''
we combine the train, dev to serve as the training vector. We have already determined the
best performing hyperparamter on the dev set, and just need to see results on test set now.
'''

x1_train_sadness = np.concatenate((x1_train_sadness,x1_dev_sadness),axis=0)

x1_test_sadness = np.load('../../intermediate_files/word2vec_based_concatenated_vectors/test/'
                   +emotion+'.npy')

print('x1_train_sadness shape:', x1_train_sadness.shape)    # (n, 50, 400)
print('x1_test_sadness shape:', x1_test_sadness.shape)      # (n, 50, 400)


# ### With reference to the Figures in our paper, the above is the first of the parallely connected components. It is the concatenated word2vec representation which can be fed to a CNN/LSTM. 
# 
# ### Below, we form the average embedding (Layer L2b) by simply taking the mean across the words of the sentence (tweet). This can then serve as input to fully connected layers (component 2 in Figure 1)

# In[ ]:

x2b_train_anger = np.mean(x1_train_anger, axis=1)
x2b_test_anger = np.mean(x1_test_anger, axis=1)

print('x2b_train_anger shape:', x2b_train_anger.shape)    # (n, 400)
print('x2b_test_anger shape:', x2b_test_anger.shape) # (n, 400)


# In[ ]:

x2b_train_fear = np.mean(x1_train_fear, axis=1)
x2b_test_fear = np.mean(x1_test_fear, axis=1)

print('x2b_train_fear shape:', x2b_train_fear.shape)    # (n, 400)
print('x2b_test_fear shape:', x2b_test_fear.shape) # (n, 400)


# In[ ]:

x2b_train_joy = np.mean(x1_train_joy, axis=1)
x2b_test_joy = np.mean(x1_test_joy, axis=1)

print('x2b_train_joy shape:', x2b_train_joy.shape)    # (n, 400)
print('x2b_test_joy shape:', x2b_test_joy.shape) # (n, 400)


# In[ ]:

x2b_train_sadness = np.mean(x1_train_sadness, axis=1)
x2b_test_sadness = np.mean(x1_test_sadness, axis=1)

print('x2b_train_sadness shape:', x2b_train_sadness.shape)    # (n, 400)
print('x2b_test_sadness shape:', x2b_test_sadness.shape) # (n, 400)


# #### We get the gold labels for our train (=train+dev) and test sets. Note that labels here means the annotated emotion intesities

# In[ ]:

emotion = 'anger'
y_train_anger = np.concatenate((np.load('../../intermediate_files/gold_label_vectors/train/'
                                 +emotion+'.npy'),
                          np.load('../../intermediate_files/gold_label_vectors/dev/'
                                 +emotion+'.npy')),axis=0)

y_test_anger = np.load('../../intermediate_files/gold_label_vectors/test/'
                                 +emotion+'.npy')
print(y_train_anger.shape)    #(n,)
print(y_test_anger.shape)     #(n,)


# In[ ]:

emotion = 'fear'
y_train_fear = np.concatenate((np.load('../../intermediate_files/gold_label_vectors/train/'
                                 +emotion+'.npy'),
                          np.load('../../intermediate_files/gold_label_vectors/dev/'
                                 +emotion+'.npy')),axis=0)

y_test_fear = np.load('../../intermediate_files/gold_label_vectors/test/'
                                 +emotion+'.npy')
print(y_train_fear.shape)    #(n,)
print(y_test_fear.shape)     #(n,)


# In[ ]:

emotion = 'joy'
y_train_joy = np.concatenate((np.load('../../intermediate_files/gold_label_vectors/train/'
                                 +emotion+'.npy'),
                          np.load('../../intermediate_files/gold_label_vectors/dev/'
                                 +emotion+'.npy')),axis=0)

y_test_joy = np.load('../../intermediate_files/gold_label_vectors/test/'
                                 +emotion+'.npy')
print(y_train_joy.shape)    #(n,)
print(y_test_joy.shape)     #(n,)


# In[ ]:

emotion = 'sadness'
y_train_sadness = np.concatenate((np.load('../../intermediate_files/gold_label_vectors/train/'
                                 +emotion+'.npy'),
                          np.load('../../intermediate_files/gold_label_vectors/dev/'
                                 +emotion+'.npy')),axis=0)

y_test_sadness = np.load('../../intermediate_files/gold_label_vectors/test/'
                                 +emotion+'.npy')
print(y_train_sadness.shape)    #(n,)
print(y_test_sadness.shape)     #(n,)


# ### Now, for the third of the parallely connected components - The deepmoji based pre-trained cnn activations (2304 dim. vector) (layer L2c) and our lexicon based features (43 dim. vector) (layer L2d). These can be produced by the corresponding subdirectories in the main directory intermediate_files/ and running the corresponding code in ../Supporting_Codes/ after that.

# In[ ]:

emotion='anger'
x2d_train_anger = np.load('../../intermediate_files/deepmoji_vectors/train/'
                                  +emotion+'.npy')
x2c_train_anger = np.load('../../intermediate_files/lexicon_vectors/train/'
                                  +emotion+'.npy')
x2d_dev_anger = np.load('../../intermediate_files/deepmoji_vectors/dev/'
                                  +emotion+'.npy')
x2c_dev_anger =  np.load('../../intermediate_files/lexicon_vectors/dev/'
                                  +emotion+'.npy')
x2d_train_anger = np.concatenate((x2d_train_anger,x2d_dev_anger),axis=0)
x2c_train_anger = np.concatenate((x2c_train_anger,x2c_dev_anger),axis=0)

x2d_test_anger = np.load('../../intermediate_files/deepmoji_vectors/test/'
                                  +emotion+'.npy')
x2c_test_anger = np.load('../../intermediate_files/lexicon_vectors/test/'
                                  +emotion+'.npy')
print('x2c_train_anger shape:', x2c_train_anger.shape)   #(n1, 43)
print('x2c_test_anger shape:', x2c_test_anger.shape)    #(n2, 43)

print('x2d_train_anger shape:', x2d_train_anger.shape)   #(n1, 2304)
print('x2d_test_anger shape:', x2d_test_anger.shape)    #(n2, 2304)


# In[ ]:

emotion='fear'
x2d_train_fear = np.load('../../intermediate_files/deepmoji_vectors/train/'
                                  +emotion+'.npy')
x2c_train_fear = np.load('../../intermediate_files/lexicon_vectors/train/'
                                  +emotion+'.npy')
x2d_dev_fear = np.load('../../intermediate_files/deepmoji_vectors/dev/'
                                  +emotion+'.npy')
x2c_dev_fear =  np.load('../../intermediate_files/lexicon_vectors/dev/'
                                  +emotion+'.npy')
x2d_train_fear = np.concatenate((x2d_train_fear,x2d_dev_fear),axis=0)
x2c_train_fear = np.concatenate((x2c_train_fear,x2c_dev_fear),axis=0)

x2d_test_fear = np.load('../../intermediate_files/deepmoji_vectors/test/'
                                  +emotion+'.npy')
x2c_test_fear = np.load('../../intermediate_files/lexicon_vectors/test/'
                                  +emotion+'.npy')
print('x2c_train_fear shape:', x2c_train_fear.shape)   #(n1, 43)
print('x2c_test_fear shape:', x2c_test_fear.shape)    #(n2, 43)

print('x2d_train_fear shape:', x2d_train_fear.shape)   #(n1, 2304)
print('x2d_test_fear shape:', x2d_test_fear.shape)    #(n2, 2304)


# In[ ]:

emotion='joy'
x2d_train_joy = np.load('../../intermediate_files/deepmoji_vectors/train/'
                                  +emotion+'.npy')
x2c_train_joy = np.load('../../intermediate_files/lexicon_vectors/train/'
                                  +emotion+'.npy')
x2d_dev_joy = np.load('../../intermediate_files/deepmoji_vectors/dev/'
                                  +emotion+'.npy')
x2c_dev_joy =  np.load('../../intermediate_files/lexicon_vectors/dev/'
                                  +emotion+'.npy')
x2d_train_joy = np.concatenate((x2d_train_joy,x2d_dev_joy),axis=0)
x2c_train_joy = np.concatenate((x2c_train_joy,x2c_dev_joy),axis=0)

x2d_test_joy = np.load('../../intermediate_files/deepmoji_vectors/test/'
                                  +emotion+'.npy')
x2c_test_joy = np.load('../../intermediate_files/lexicon_vectors/test/'
                                  +emotion+'.npy')
print('x2c_train_joy shape:', x2c_train_joy.shape)   #(n1, 43)
print('x2c_test_joy shape:', x2c_test_joy.shape)    #(n2, 43)

print('x2d_train_joy shape:', x2d_train_joy.shape)   #(n1, 2304)
print('x2d_test_joy shape:', x2d_test_joy.shape)    #(n2, 2304)


# In[ ]:

emotion='sadness'
x2d_train_sadness = np.load('../../intermediate_files/deepmoji_vectors/train/'
                                  +emotion+'.npy')
x2c_train_sadness = np.load('../../intermediate_files/lexicon_vectors/train/'
                                  +emotion+'.npy')
x2d_dev_sadness = np.load('../../intermediate_files/deepmoji_vectors/dev/'
                                  +emotion+'.npy')
x2c_dev_sadness =  np.load('../../intermediate_files/lexicon_vectors/dev/'
                                  +emotion+'.npy')
x2d_train_sadness = np.concatenate((x2d_train_sadness,x2d_dev_sadness),axis=0)
x2c_train_sadness = np.concatenate((x2c_train_sadness,x2c_dev_sadness),axis=0)

x2d_test_sadness = np.load('../../intermediate_files/deepmoji_vectors/test/'
                                  +emotion+'.npy')
x2c_test_sadness = np.load('../../intermediate_files/lexicon_vectors/test/'
                                  +emotion+'.npy')
print('x2c_train_sadness shape:', x2c_train_sadness.shape)   #(n1, 43)
print('x2c_test_sadness shape:', x2c_test_sadness.shape)    #(n2, 43)

print('x2d_train_sadness shape:', x2d_train_sadness.shape)   #(n1, 2304)
print('x2d_test_sadness shape:', x2d_test_sadness.shape)    #(n2, 2304)


# ## Set the hyperparameters (set to our optimal ones)

# In[ ]:

hyperparams = {'l2a': 250, 'l3a': 128, 'l3b': 256, 'l5a': 64, 'l6a': 32, 'l5f': 75, 'l6f': 40, 'l5j': 64, 'l6j': 32, 'l5s': 40, 'l6s': 20}


# In[ ]:

x_train_anger = [x1_train_anger, x2b_train_anger, x2c_train_anger, x2d_train_anger]
x_train_fear = [x1_train_fear, x2b_train_fear, x2c_train_fear, x2d_train_fear]
x_train_joy = [x1_train_joy, x2b_train_joy, x2c_train_joy, x2d_train_joy]
x_train_sadness = [x1_train_sadness, x2b_train_sadness, x2c_train_sadness, x2d_train_sadness]


# In[ ]:

x_test_anger = [x1_test_anger, x2b_test_anger, x2c_test_anger, x2d_test_anger]
x_test_fear = [x1_test_fear, x2b_test_fear, x2c_test_fear, x2d_test_fear]
x_test_joy = [x1_test_joy, x2b_test_joy, x2c_test_joy, x2d_test_joy]
x_test_sadness = [x1_test_sadness, x2b_test_sadness, x2c_test_sadness, x2d_test_sadness]


# In[ ]:

# To account for variations in results when models like CNN are involved, you could run 
# the code 7 times and take average of results as we did in LE-PC-DNN code.

#component1
l1 = Input(shape=(50,400,))

l2a = Conv1D(hyperparams['l2a'], 3, activation='relu')(l1)
l2a = GlobalMaxPooling1D()(l2a)
l2a = Dropout(p=0.2)(l2a)
l3a = Dense(hyperparams['l3a'], activation='relu')(l2a)

#component2
l2b = Input(shape=(400,))
l3b = Dropout(p=0.2)(l2b)
l3b = Dense(hyperparams['l3b'], activation='relu')(l3b)

#component3
l2c = Input(shape=(43,))
l2d = Input(shape=(2304,))

#merge, component4
l4 = merge([l3a, l3b, l2c, l2d], mode='concat', concat_axis=-1)

#separation of network for different emotions
#anger
l5a = Dense(hyperparams['l5a'], activation='relu')(l4)
l6a = Dense(hyperparams['l6a'], activation='relu')(l5a)
output_anger = Dense(1, activation='sigmoid')(l6a)

#fear
l5f = Dense(hyperparams['l5f'], activation='relu')(l4)
l6f = Dense(hyperparams['l6f'], activation='relu')(l5f)
output_fear = Dense(1, activation='sigmoid')(l6f)

#joy
l5j = Dense(hyperparams['l5j'], activation='relu')(l4)
l6j = Dense(hyperparams['l6j'], activation='relu')(l5j)
output_joy = Dense(1, activation='sigmoid')(l6j)

#sadness
l5s = Dense(hyperparams['l5s'], activation='relu')(l4)
l6s = Dense(hyperparams['l6s'], activation='relu')(l5s)
output_sadness = Dense(1, activation='sigmoid')(l6s)

inp = [l1, l2b, l2c, l2d]
model_anger = Model(input=inp, output=output_anger)
model_anger.compile(loss='mae', optimizer='adam', metrics=['mae'])
model_fear = Model(input=inp, output=output_fear)
model_fear.compile(loss='mae', optimizer='adam', metrics=['mae'])
model_joy = Model(input=inp, output=output_joy)
model_joy.compile(loss='mae', optimizer='adam', metrics=['mae'])
model_sadness = Model(input=inp, output=output_sadness)
model_sadness.compile(loss='mae', optimizer='adam', metrics=['mae'])

# To time the architecture
#start = timeit.default_timer()

batch_size = 32
nb_epoch=40
n_batches_anger = x_train_anger[0].shape[0]//batch_size
n_batches_fear = x_train_fear[0].shape[0]//batch_size
n_batches_joy = x_train_joy[0].shape[0]//batch_size
n_batches_sadness = x_train_sadness[0].shape[0]//batch_size

for iepoch in range(nb_epoch):
    print('Starting epoch :', iepoch+1)

    for ibatch in range(n_batches_anger) :
        xbatch = [x_train_anger[0][ibatch*batch_size : (ibatch+1)*batch_size],
                  x_train_anger[1][ibatch*batch_size : (ibatch+1)*batch_size],
                  x_train_anger[2][ibatch*batch_size : (ibatch+1)*batch_size],
                  x_train_anger[3][ibatch*batch_size : (ibatch+1)*batch_size]]
        ybatch = y_train_anger[ibatch*batch_size : (ibatch+1)*batch_size]
        model_anger.train_on_batch(xbatch,ybatch)

    for ibatch in range(n_batches_fear) :
        xbatch = [x_train_fear[0][ibatch*batch_size : (ibatch+1)*batch_size],
                  x_train_fear[1][ibatch*batch_size : (ibatch+1)*batch_size],
                  x_train_fear[2][ibatch*batch_size : (ibatch+1)*batch_size],
                  x_train_fear[3][ibatch*batch_size : (ibatch+1)*batch_size]]
        ybatch = y_train_fear[ibatch*batch_size : (ibatch+1)*batch_size]
        model_fear.train_on_batch(xbatch,ybatch)

    for ibatch in range(n_batches_joy) :
        xbatch = [x_train_joy[0][ibatch*batch_size : (ibatch+1)*batch_size],
                  x_train_joy[1][ibatch*batch_size : (ibatch+1)*batch_size],
                  x_train_joy[2][ibatch*batch_size : (ibatch+1)*batch_size],
                  x_train_joy[3][ibatch*batch_size : (ibatch+1)*batch_size]]
        ybatch = y_train_joy[ibatch*batch_size : (ibatch+1)*batch_size]
        model_joy.train_on_batch(xbatch,ybatch)

    for ibatch in range(n_batches_sadness) :
        xbatch = [x_train_sadness[0][ibatch*batch_size : (ibatch+1)*batch_size],
                  x_train_sadness[1][ibatch*batch_size : (ibatch+1)*batch_size],
                  x_train_sadness[2][ibatch*batch_size : (ibatch+1)*batch_size],
                  x_train_sadness[3][ibatch*batch_size : (ibatch+1)*batch_size]]
        ybatch = y_train_sadness[ibatch*batch_size : (ibatch+1)*batch_size]
        model_sadness.train_on_batch(xbatch,ybatch)
                    
# stop = timeit.default_timer()
# print('Time taken to train LE-PC-DMTL: ', stop-start)

'''
You can check the number of trainable parameters by printing model summary as in comments
below.
'''
# print('Network parameters for anger: ', model_anger.summary())

# print('Network parameters for fear: ', model_fear.summary())

# print('Network parameters for joy: ', model_joy.summary())

# print('Network parameters for sadness: ', model_sadness.summary())

y_pred_anger = model_anger.predict(x_test_anger)
y_pred_fear = model_fear.predict(x_test_fear)
y_pred_joy = model_joy.predict(x_test_joy)
y_pred_sadness = model_sadness.predict(x_test_sadness)


# In[ ]:

pearson_correlation_score_anger = pearsonr(y_pred_anger.reshape((y_pred_anger.shape[0],))
                                           , y_test_anger)[0]

print('Pearson Correlation for LE_PC_DMTL model on Test set for anger')
print(pearson_correlation_score_anger)


# In[ ]:

pearson_correlation_score_fear = pearsonr(y_pred_fear.reshape((y_pred_fear.shape[0],))
                                           , y_test_fear)[0]

print('Pearson Correlation for LE_PC_DMTL model on Test set for fear')
print(pearson_correlation_score_fear)


# In[ ]:

pearson_correlation_score_joy = pearsonr(y_pred_joy.reshape((y_pred_joy.shape[0],))
                                           , y_test_joy)[0]

print('Pearson Correlation for LE_PC_DMTL model on Test set for joy')
print(pearson_correlation_score_joy)


# In[ ]:

pearson_correlation_score_sadness = pearsonr(y_pred_sadness.reshape((y_pred_sadness.shape[0],))
                                           , y_test_sadness)[0]

print('Pearson Correlation for LE_PC_DMTL model on Test set for sadness')
print(pearson_correlation_score_sadness)

