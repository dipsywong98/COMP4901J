import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
from cs231n.data_utils import load_CIFAR10
# %matplotlib inline

def run_model(session, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False):
    # have tensorflow compute accuracy
    prediction = tf.argmax(predict,1)
    correct_prediction = tf.equal(prediction, y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    # np.random.shuffle(train_indicies)

    training_now = training is not None
    
    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [mean_loss,correct_prediction,accuracy,prediction]
    if training_now:
        variables[-1] = training
    _pre = np.array([])
    # counter 
    iter_cnt = 0
    for e in range(epochs):
        # keep track of losses and accuracy
        correct = 0
        losses = []
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
            # generate indicies for the batch
            start_idx = (i*batch_size)%Xd.shape[0]
            idx = train_indicies[start_idx:start_idx+batch_size]
            
            # create a feed dictionary for this batch
            feed_dict = {X: Xd[idx,:],
                         y: yd[idx],
                         is_training: training_now }
            # get batch size
            actual_batch_size = yd[idx].shape[0]
            
            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, corr, _, __pre = session.run(variables,feed_dict=feed_dict)
            # print(_pre,__pre)
            _pre = np.concatenate((_pre,__pre))
            
            # aggregate performance stats
            losses.append(loss*actual_batch_size)
            correct += np.sum(corr)
            
            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
                      .format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
                if np.sum(corr)/actual_batch_size >= 0.98:
                    break #the model reach the trainning optimal
            iter_cnt += 1
        total_correct = correct/Xd.shape[0]
        total_loss = np.sum(losses)/Xd.shape[0]
        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\
              .format(total_loss,total_correct,e+1))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e+1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()
    return total_loss,total_correct,_pre

def my_model(X,y,is_training):
    # con bn pool con bn pool affine
    conv1 = tf.nn.conv2d(X, Wconv1, strides=[1,1,1,1], padding='SAME') + bconv1
    h_batch1 = tf.layers.batch_normalization(conv1, center=True, scale=True,  training=is_training)
    h1 = tf.nn.relu(h_batch1)
    h_pool1 = tf.nn.max_pool(h1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    conv2 = tf.nn.conv2d(h_pool1, Wconv2, strides=[1,1,1,1], padding='SAME') + bconv2
    h_batch2 = tf.layers.batch_normalization(conv2, center=True, scale=True,  training=is_training)
    h2 = tf.nn.relu(h_batch2)
    h_pool2 = tf.nn.max_pool(h2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    conv_flat = tf.reshape(h_pool2,[-1,4096])
    affine = tf.matmul(conv_flat,W1) + b1
    h_batch3 = tf.layers.batch_normalization(affine, center=True, scale=True,  training=is_training)
    h3 = tf.nn.relu(h_batch3)
    y_out = tf.matmul(h3,W2) + b2
    return y_out

tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)
global_step = tf.Variable(0)

Wconv1 = tf.get_variable("Wconv1", shape=[5, 5, 3, 32])
bconv1 = tf.get_variable("bconv1", shape=[32])
Wconv2 = tf.get_variable("Wconv2", shape=[5, 5, 32, 64])
bconv2 = tf.get_variable("bconv2", shape=[64])

W1 = tf.get_variable("W1", shape=[4096, 1024])
b1 = tf.get_variable("b1", shape=[1024])
W2 = tf.get_variable("W2", shape=[1024, 10])
b2 = tf.get_variable("b2", shape=[10])
    
    
y_out_multiple = my_model(X,y,is_training)
mean_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y,10), logits=y_out))
mean_loss += 0.001 * (tf.nn.l2_loss(Wconv1) + tf.nn.l2_loss(Wconv2) + tf.nn.l2_loss(W1))

learning_rate = tf.train.exponential_decay(0.001, global_step, 100, 0.9, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate)

# batch normalization in tensorflow requires this extra dependency
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_step = optimizer.minimize(mean_loss, global_step=global_step)
