from tqdm import tqdm
import numpy as np
from collections import Counter
import random

with open('test_clip_length.txt','r') as f:
    line = f.readline().split()
test_clip_length = [int(x) for x in line]

with open('test_preds.txt','r') as f:
    line = f.readline().split()
test_preds = [int(x) for x in line]

with open('test_labels.txt','r') as f:
    line = f.readline().split()
test_labels = [int(x) for x in line]

with open('test_logits.txt','r') as f:
    line = f.readlines()
test_logits = np.asarray([[float(y) for y in x.strip().split()] for x in line])

def confusion_matrix(label,pred,num_classes=2):
    hist = np.zeros((num_classes,num_classes))
    for i,j in zip(label,pred):
        hist[i,j] += 1
    axis_sum = np.sum(hist,axis=1)
    return hist

y_clip_label = []
abs_cnt,standard_cnt,logits_cnt = [],[],[]
for i in tqdm(range(len(test_clip_length)-1)):
    start,end = test_clip_length[i],test_clip_length[i+1]
    y_labels = test_labels[start:end]
    y_preds = test_preds[start:end]
    y_logits = test_logits[start:end]
    
    clip_label = Counter(y_labels).most_common(1)[0][0]
    y_clip_label.append(clip_label)

    y_pred_max_count = Counter(y_preds).most_common(1)[0][1]
    y_pred_dict = dict(Counter(y_preds))

    # List of the classes that are predicted maximum times.
    # For eg. if predictions = [1,1,2,3,4,3,2,1,2] 
    # max_freq = [1,2] as both of them occur maximum(3) 
    # times in predictions.

    max_freq = []
    for key in list(y_pred_dict.keys()):
        if y_pred_dict[key] == y_pred_max_count:
            max_freq.append(key)

    max_freq = list(set(list(max_freq)))
    # print(clip_label,max_freq[0],Counter(y_labels),Counter(y_preds))

    # Absolute Clip Accuracy (Strictest Accuracy): If the model predicts only one 
    # class for each and every frame of a particular clip then 
    # only we assume that the model has predicted the class of the clip.
    # Note that this can't predict for all clips. For e.g. if 8/10 frames of a clip are predicted 1 and rest are predicted zero, we can't predict any class.
    abs_cnt.append(1) if clip_label == max_freq[0] and len(Counter(y_preds)) == 1 else abs_cnt.append(0)

    # Standard Clip Accuracy: If len(max_freq) == 1 i.e. there is exactly one 
    # class with maximum frequency and that class is the clip label, we assume 
    # the model has correctly predicted the class of the clip. Statistically 
    # speaking if the clip label is the mode of the predictions.For example,there is a 
    # 5 second clip of class 1 and the model predicts = [1,1,2,3,0] then we say 
    # that the model has predicted correct. If it predicts [1,1,2,2,3], we say it is incorrect.
    standard_cnt.append(max_freq[0]) 

    # Average Logits Accuracy
    logits_cnt.append(np.argmax(np.mean(y_logits,axis=0)))

# Frame Level Accuracy. 
frame_cnt = [1 for x,y in zip(test_labels,test_preds) if x == y]

print('frame_accuracy:',np.sum(frame_cnt)/float(len(test_labels)))
print('absolute_clip_accuracy:',np.sum(abs_cnt)/float(len(test_clip_length)-1))
print('standard_clip_accuracy:',np.sum(np.asarray(standard_cnt) == np.asarray(y_clip_label))/float(len(test_clip_length)-1))
print('average_logits_clip_accuracy:',np.sum(np.asarray(logits_cnt) == np.asarray(y_clip_label))/float(len(test_clip_length)-1))
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(confusion_matrix(y_clip_label,standard_cnt))
