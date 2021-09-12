"""

Awesome NN exercises by Shivendra Kumar

"""

import pandas as pd
import numpy as np
import tensorflow as tf
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score as auc

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

data = pd.read_csv('/Users/sumogroup/shivendra/BaseAnacondaProjs/autoEncoder/creditcard.csv')
print(data.shape)

#Lets visualize the columns
print(data.columns)
print(data.dtypes)

#Lets view part of the data
print(data.head(5))

#Lets look at the fradulent  cases
print("{:.3f} % of all transaction are fraud. ".format(np.sum(data['Class'])/data.shape[0]*100))
print("{}  transaction are fraud. ".format(np.sum(data['Class'])))
#Lets look at the spread of data across fradulent Vs non Fraud
plt.figure(figsize=(12,5*4))
gs = gridspec.GridSpec(5,1)
for i, cn in enumerate(data.columns[10:15]):
    ax = plt.subplot(gs[i])
    sns.distplot(data[cn][data.Class==1], bins=50)
    sns.distplot(data[cn][data.Class==0], bins=50)
    ax.set_xlabel('')
    ax.set_title('histograms of feature: '+ str(cn))
plt.show


#Training and validation set preparation
TEST_RATIO = 0.20
data.sort_values('Time', inplace= True)
TRA_INDEX = int((1-TEST_RATIO)*data.shape[0])
train_x = data.iloc[:TRA_INDEX, 1:-2 ].values
train_y = data.iloc[:TRA_INDEX, -1].values
test_x = data.iloc[TRA_INDEX:, 1:-2 ].values
test_y = data.iloc[TRA_INDEX:, -1].values


print("Total Training examples: {}, Total Fradulent Cases: {}, percentage of total {:.5f}".format(train_x.shape[0], np.sum(train_y), 
      (np.sum(train_y)/train_x.shape[0])*100))
print("Total Testing examples: {}, Total Fradulent Cases: {}, percentage of total {:.5f}".format(test_x.shape[0], np.sum(test_y), 
      (np.sum(test_y)/test_y.shape[0])*100))
    

cols_mean = []
cols_std = []

for c in range(train_x.shape[1]):
    cols_mean.append(train_x[:,c].mean())
    cols_std.append(train_x[:,c].std())
    train_x[:,c] = (train_x[:,c] - cols_mean[-1])/cols_std[-1]
    test_x[:,c] = (test_x[:,c] - cols_mean[-1])/cols_std[-1]
    
    
learning_rate = 0.0014
training_epochs = 1000    
batch_size = 256
display_step = 10
n_hidden1 = 15
n_input = train_x.shape[1]

X = tf.placeholder( "float", [None , n_input])

weights = {
        'encoder_h1' : tf.Variable(tf.random_normal([n_input, n_hidden1])),
        'decoder_h1' : tf.Variable(tf.random_normal([n_hidden1, n_input])),
        }

biases = {
        'encoder_b1' : tf.Variable(tf.random_normal([n_hidden1])),
        'decoder_b1' : tf.Variable(tf.random_normal([n_input])),
        }

def encoder(x):
    layer1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    return layer1

def decoder(x):
    layer1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    return layer1


encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

y_pred = decoder_op
y_true = X
batch_mse = tf.reduce_mean(tf.pow(y_true-y_pred, 2),1)

cost_op = tf.reduce_mean(tf.pow(y_true-y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost_op)

save_model  = '/Users/apple/AnacondaProjects/autoEncoder/data/autoEinterme.ckpt'
saver = tf.train.Saver()

init_op = tf.global_variables_initializer() 

epoch_list = []
loss_list = []
train_auc_list = []

data_dir = 'data'

with tf.Session() as sess:
    now = datetime.now()
    sess.run(init_op)
    total_batch = int(train_x.shape[0]/batch_size)
    
    #Lets start the training
    
    for epoch in range(training_epochs):
        
        for i in range(total_batch):
            batch_idx = np.random.choice(train_x.shape[0], batch_size)
            batch_xs = train_x[batch_idx]
            
            #Run optimization op
            _,c = sess.run([optimizer, cost_op], feed_dict = {X: batch_xs})
            
        # Display  logs per epoch
        if epoch % display_step == 0:
            train_batch_mse = sess.run( batch_mse, feed_dict = {X:train_x})
            epoch_list.append(epoch+1)
            loss_list.append(c)
            train_auc_list.append(auc(train_y, train_batch_mse))
            print("Epoch :", '%04d,', (epoch+1),
                  "Cost : ", "{:.9f},".format(c),
                  "Train_auc=", "{:.6f},".format(auc(train_y, train_batch_mse)),)
                  
             
    print("Optimization finished! ")
    save_path = saver.save(sess, save_model)
    print("Model Saved  in: %s " % save_path)
     
save_model = '/Users/apple/AnacondaProjects/autoEncoder/data/AutoEfinal.ckpt'
saver = tf.train.Saver()
     
#Plot Training AUC over time 
plt.plot(epoch_list,   train_auc_list, 'b--', label='Training AUC', linewidth=1.0)
plt.title('Training AUC per iteration')
plt.xlabel('Iteration')    
plt.ylabel('Training AUC')
plt.legend(loc='upper right')
plt.grid(True)

# Lets plot the training loss over time
plt.plot(epoch_list,   loss_list, 'r--', label='Training loss', linewidth=1.0)
plt.title('Training Loss')
plt.xlabel('Iteration')    
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

#We trained an Auto encoder to detect fradulent transactions
#Lets check its accuracy on a test dataset

save_model = '/Users/apple/AnacondaProjects/autoEncoder/data/autoEinterme.ckpt'
saver = tf.train.Saver()

#Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    now = datetime.now()
    saver.restore(sess, save_model)
    test_batch_mse = sess.run(batch_mse, feed_dict= {X:test_x})
    print("test AUC score: {:.6f}".format(auc(test_y, test_batch_mse)))
   

    
    
    