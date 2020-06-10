# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import print_function

from random import shuffle
import random
import datetime

import numpy as np
import tensorflow as tf
import h5py
from tqdm import tqdm

import sys
sys.path.append('..')
import vggish_params
import vggish_slim

flags = tf.app.flags
slim = tf.contrib.slim

flags.DEFINE_integer(
    'num_units', 100,
    'Number of units of examples to feed into the model.'
    )

flags.DEFINE_boolean(
    'train_vggish', True,
    'If True, allow VGGish parameters to change during training, thus '
    'fine-tuning VGGish. If False, VGGish parameters are fixed, thus using '
    'VGGish as a fixed feature extractor.')

flags.DEFINE_string(
    'checkpoint', '../vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

FLAGS = flags.FLAGS
shuffle_train = True

_NUM_EPOCHS = 10000
_NUM_CLASSES = 2
_BATCH_SIZE = 16
_PATIENCE = 5
_LIMIT = 0.01


train_filename = 'train'
if shuffle_train:
    train_filename = train_filename + '_shuffled'
train_filename = train_filename + '.hdf5'

f = h5py.File(train_filename,'r')
train_data,train_label = f['train_data'],f['train_label']
labeled_data = list(zip(train_data,train_label))

_NUM_BATCHES = int(np.ceil(f['train_data'].shape[0]/float(_BATCH_SIZE)))

def _get_examples_batch(x,labeled_data,batch_size=_BATCH_SIZE):
  """Returns a shuffled batch of examples of all audio classes.

  Returns:
    a tuple (features, labels) where features is a NumPy array of shape
    [batch_size, num_frames, num_bands] where the batch_size is variable and
    each row is a log mel spectrogram patch of shape [num_frames, num_bands]
    suitable for feeding VGGish, while labels is a NumPy array of shape
    [batch_size, num_classes] where each row is a multi-hot label vector that
    provides the labels for corresponding rows in features.
  """
  batch_labeled_data = labeled_data[x*batch_size:min((x+1)*batch_size,len(labeled_data))]
  features = [example for (example, _) in batch_labeled_data]
  labels = [label for (_, label) in batch_labeled_data]
  return (features, labels)

f2 = h5py.File('val.hdf5','r')
val_data,val_label = f2['val_data'],f2['val_label']
val_labeled_data = list(zip(val_data,val_label))

f3 = h5py.File('test.hdf5','r')
test_data,test_label = f3['test_data'],f3['test_label']
test_labeled_data = list(zip(test_data,test_label))

# Define VGGish.
embeddings = vggish_slim.define_vggish_slim(FLAGS.train_vggish)

# Define a shallow classification model and associated training ops on top of VGGish.
# Add a fully connected layer with FLAGS.num_units units.
num_units = FLAGS.num_units

fc = slim.fully_connected(embeddings, num_units)

# Add a classifier layer at the end, consisting of parallel logistic classifiers, one per class. This allows for multi-class tasks.
logits = slim.fully_connected(fc, _NUM_CLASSES, activation_fn=None, scope='logits')
# logits = tf.sigmoid(logits, name='prediction')

# Add training ops.
global_step = tf.Variable(0, name='global_step', trainable=False,collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                 tf.GraphKeys.GLOBAL_STEP])
# Labels are assumed to be fed as a batch multi-hot vectors, with a 1 in the position of each positive class label, and 0 elsewhere.
labels = tf.placeholder(tf.float32, shape=(None, _NUM_CLASSES), name='labels')

# Cross-entropy label loss.
xent = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels, name='xent')
loss = tf.reduce_mean(xent, name='loss_op')
tf.summary.scalar('loss', loss)

# We use the same optimizer and hyperparameters as used to train VGGish.
optimizer = tf.train.AdamOptimizer(learning_rate=vggish_params.LEARNING_RATE,epsilon=vggish_params.ADAM_EPSILON).minimize(loss, global_step=global_step, name='train_op')

# Saver object to save the model.
saver = tf.train.Saver()

# Initialize all variables in the model, and then load the pre-trained
# VGGish checkpoint.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)

    # Locate all the tensors and ops we need for the training loop.
    features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
    # labels_tensor = sess.graph.get_tensor_by_name('mymodel/train/labels:0')
    # global_step_tensor = sess.graph.get_tensor_by_name('mymodel/train/global_step:0')
    # loss_tensor = sess.graph.get_tensor_by_name('mymodel/train/loss_op:0')
    # train_op = sess.graph.get_operation_by_name('mymodel/train/train_op')
    # saver.restore(sess,'./best_model_400_True.ckpt')
    # print('Model Restored.')

    # The training loop.
    accuracy = []
    min_val_loss = 1000000
    best_val_acc = 0.0
    early_stopping = 0
    for k in range(_NUM_EPOCHS): 
        batch_loss = 0.0
        for i in tqdm(range(_NUM_BATCHES)):
            (batch_features, batch_labels) = _get_examples_batch(i,labeled_data)
            _,_loss = sess.run([optimizer, loss],\
                feed_dict={features_tensor: batch_features,labels: batch_labels})
            # print('Epoch # %d, Step %d,  loss %g ' % (k+1,i+1,_loss))
            batch_loss += _loss
        batch_loss /= float(_NUM_BATCHES)
        print('Epoch # %d, train_loss: %g'%(k+1,batch_loss))
        
        if (k % 10 == 0 and k > 0) or k == _NUM_EPOCHS:  
            (val_features,val_label) = _get_examples_batch(0,val_labeled_data,\
                                                batch_size=len(val_labeled_data))
            val_loss = sess.run(loss,feed_dict={features_tensor:val_features,labels:val_label})

            preds = logits.eval(feed_dict={features_tensor:val_features})
            pred_label = np.argmax(preds,1)
            val_correct = np.asarray([1 for x,y in zip(pred_label,np.argmax(val_label,1)) if x == y])
            val_acc = np.sum(val_correct)/float(len(val_labeled_data))

            print('Epoch # %d, val_loss: %g, val_acc: %g' %(k+1,val_loss,val_acc))
            
            if val_loss < min_val_loss +_LIMIT and val_acc > best_val_acc:
                early_stopping = 0
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                best_val_acc = val_acc
                model_name = 'best_model_'+str(FLAGS.num_units)+'_'+str(FLAGS.train_vggish)
                model_name = './'+model_name+'.ckpt'
                saver.save(sess,model_name)
                print('Saving model as best_model_%d_%s.ckpt after %d epochs.'\
                                %(FLAGS.num_units,FLAGS.train_vggish,k+1))
            else:
                early_stopping += 1
            
            if early_stopping == _PATIENCE:
                break
    
    saver.restore(sess,model_name)
    print('Model loaded for testing.')
    (test_features,test_label) = _get_examples_batch(0,test_labeled_data,\
                                                batch_size=len(test_labeled_data)) 
    test_loss = sess.run(loss,feed_dict={features_tensor:test_features,labels:test_label})

    test_preds = logits.eval(feed_dict={features_tensor:test_features})
    test_pred_label = np.argmax(test_preds,1)
    test_correct = np.asarray([1 for x,y in zip(test_pred_label,np.argmax(test_label,1)) if x == y])
    test_acc = np.sum(test_correct)/float(len(test_labeled_data))

    with open('test_logits.txt','a') as f:
        f.write(str(FLAGS.num_units))
        f.write('\n------------------------------------------------------------\n')
        for logit in test_preds:
            f.write(' '.join([str(x) for x in logit]))
            f.write('\n')
        f.write('\n------------------------------------------------------------\n')
    with open('test_preds.txt','a') as f1:
        f1.write(str(FLAGS.num_units))
        f1.write('\n------------------------------------------------------------\n')
        f1.write(' '.join([str(x) for x in np.argmax(test_preds,1)]))
        f1.write('\n------------------------------------------------------------\n')
    with open('test_labels.txt','a') as f2:
        f2.write('\n------------------------------------------------------------\n')
        f2.write(' '.join([str(x) for x in np.argmax(test_label,1)]))
        f2.write('\n------------------------------------------------------------\n')
    print('test_loss: %g, test_acc: %g' %(test_loss,test_acc))

    with open('test_results.txt','a') as f:
        f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + '\t' + \
                'model_num_units_'+ str(FLAGS.num_units) + '_train_' + str(FLAGS.train_vggish)\
                    + ' test_acc:' + str(test_acc) + ' ' + 'val_acc:' + str(best_val_acc) + '\n')
    print('Finished Training. Results appended to log file.')
