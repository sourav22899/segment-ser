import numpy as np
import os
import h5py
from tqdm import tqdm
import random 
import csv

import sys
sys.path.append('..')
import mel_features
import vggish_params
import vggish_input

shuffle_data = True
shuffle_train = False
train_split = 0.8
val_split  = 0.1
test_split = 0.1
num_classes = 4

# File search utility function.

def file_search(dirname, ret, list_avoid_dir=[]):    
    filenames = os.listdir(dirname)    
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        if os.path.isdir(full_filename) :
            if full_filename.split('/')[-1] in list_avoid_dir:
                continue
            else:
                file_search(full_filename, ret, list_avoid_dir)           
        else:
            ret.append( full_filename )          


# Gets the list of files using file_search utility

list_files = []

for x in range(5):
    sess_name = 'Session' + str(x+1)
    path = '/home/pkumar99/iit-roorkee/multimodal-speech-emotion/data/raw/IEMOCAP_full_release/' + sess_name + '/sentences/wav/'
    file_search(path, list_files)
    list_files = sorted(list_files)
    print(sess_name + ", #sum files: " + str(len(list_files)))

# Make a dictionary of type 'wav_file':'emotion'.

with open('/home/pkumar99/iit-roorkee/multimodal-speech-emotion/data/processed/IEMOCAP/label.csv') as f :
    csv_reader = csv.reader(f)
    lines = [x for x in csv_reader]

raw_data = {}
for line in lines:
    raw_data[line[0]] = line[1]

# Filter out all the emotions that are not considered.

emotion_list = ['ang','hap','exc','sad','neu']
data = {}
for key in list(raw_data.keys()):
    if raw_data[key] in emotion_list:
        data[key] = raw_data[key]

# Shuffle data if required.

random.seed(221262)

if shuffle_data:
    keys = list(data.keys()) 
    random.shuffle(keys)
    # [(key,data[key]) for key in list(data.keys())]

# Generate test/val/train data

n_keys = len(keys)
train_key_dict,val_key_dict,test_key_dict = {},{},{}

for key in keys[:int(train_split*n_keys)]:
    train_key_dict[key] = True
for key in keys[int(train_split*n_keys):int((train_split+val_split)*n_keys)]:
    val_key_dict[key] = True
for key in keys[int((train_split+val_split)*n_keys):]:
    test_key_dict[key] = True

# print(len(train_key_dict),len(val_key_dict),len(test_key_dict))

# Assign a number to each of the different emotions

list_category = [
                'ang',
                'hap',
                'sad',
                'neu',
                'fru',
                'exc',
                'fea',
                'sur',
                'dis',
                'oth',
                'xxx'
                ]

category = {}
for c_type in list_category:
    if c_type in category:
        ("")
    else:
        category[c_type] = len(category)

# Excited is considered the same class as happiness. Modify this segment when using one vs all classifier.

category['exc'] = 1
 
emotions = [0]*num_classes
test_clip_length = [0]
train_data,val_data,test_data = [],[],[]
train_label,val_label,test_label = [],[],[]
for i in tqdm(range(len(list_files))):
    suffix = list_files[i].split('/')[-1][:-4]
    if suffix in train_key_dict or suffix in val_key_dict or suffix in test_key_dict:
        one_hot_label = [0]*num_classes

        original_length_in_samples,length,spectrogram = vggish_input.wavfile_to_examples(list_files[i])
        emotions[category[data[suffix]]] += spectrogram.shape[0]
        length = int(length/16000.0)

        if suffix in train_key_dict:
            train_data.extend(spectrogram)
            one_hot_label[category[data[suffix]]] = 1
            train_label.extend([one_hot_label]*spectrogram.shape[0])

        elif suffix in val_key_dict:
            val_data.extend(spectrogram)
            one_hot_label[category[data[suffix]]] = 1
            val_label.extend([one_hot_label]*spectrogram.shape[0])
            
        elif suffix in test_key_dict:
            test_data.extend(spectrogram)
            one_hot_label[category[data[suffix]]] = 1
            test_label.extend([one_hot_label]*spectrogram.shape[0])
            test_clip_length.append(test_clip_length[-1]+spectrogram.shape[0])

if shuffle_train:
    labelled_train_data = list(zip(train_data,train_label))
    random.seed(221262)
    random.shuffle(labelled_train_data)
    train_data = [data for (data,_) in labelled_train_data]
    train_label = [label for (_,label) in labelled_train_data]

train_data,val_data,test_data = np.asarray(train_data),np.asarray(val_data),np.asarray(test_data)
train_label,val_label,test_label = np.asarray(train_label),np.asarray(val_label),np.asarray(test_label)

print(np.sum(train_label),np.sum(val_label),np.sum(test_label))
print(train_label.shape,val_label.shape,test_label.shape)
print(emotions)

with open('test_clip_length.txt','w') as f:
    f.write(' '.join([str(x) for x in test_clip_length]))

# Create hdf5 files to store test/val/test data.

train_filename = 'train'
if shuffle_train:
    train_filename = train_filename + '_shuffled'
train_filename = train_filename + '.hdf5'

train_hdf5 = h5py.File(train_filename,'w')
train_hdf5.create_dataset('train_data',data=train_data)
train_hdf5.create_dataset('train_label',data=train_label)
train_hdf5.close()

val_hdf5 = h5py.File('val.hdf5','w')
val_hdf5.create_dataset('val_data',data=val_data)
val_hdf5.create_dataset('val_label',data=val_label)
val_hdf5.close()

test_hdf5 = h5py.File('test.hdf5','w')
test_hdf5.create_dataset('test_data',data=test_data)
test_hdf5.create_dataset('test_label',data=test_label)
test_hdf5.close()