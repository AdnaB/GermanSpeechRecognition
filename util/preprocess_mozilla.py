from pydub import AudioSegment
import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import scipy.io.wavfile as wav
from python_speech_features import logfbank

import argparse
import csv

# parser = argparse.ArgumentParser(description='Mozilla German preprocess.')
#
# parser.add_argument('root', metavar='root', type=str,
#                      help='Absolute file path to Mozilla German. (e.g. /usr/downloads/LibriSpeech/)')
#
# parser.add_argument('tr_sets', metavar='tr_sets', type=str, nargs='+',
#                      help='Training datasets to process in Mozilla (e.g. train-clean-100/)')
#
# parser.add_argument('--dev_sets', metavar='dev_sets', type=str, nargs='+', default=[] ,
#                      help='Validation datasets to process in Mozilla. (e.g. dev-clean/)')
#
# parser.add_argument('--tt_sets', metavar='tt_sets', type=str, nargs='+', default=[] ,
#                      help='Testing datasets to process in Mozilla. (e.g. test-clean/)')
#
# parser.add_argument('--n_jobs', dest='n_jobs', action='store', default=-2 ,
#                    help='number of cpu availible for preprocessing.\n -1: use all cpu, -2: use all cpu but one')
# parser.add_argument('--n_filters', dest='n_filters', action='store', default=40 ,
#                    help='number of filters for fbank. (Default : 40)')
# parser.add_argument('--win_size', dest='win_size', action='store', default=0.025 ,
#                    help='window size during feature extraction (Default : 0.025 [25ms])')
# parser.add_argument('--norm_x', dest='norm_x', action='store', default=False ,
#                    help='Normalize features s.t. mean = 0 std = 1')
#
# paras = parser.parse_args()

# root = paras.root
# train_path = paras.tr_sets
# dev_path = paras.dev_sets
# test_path = paras.tt_sets
# n_jobs = paras.n_jobs
# n_filters = paras.n_filters
# win_size = paras.win_size
# norm_x = paras.norm_x
# /home/sanne/Documents/RUG/DeepLearning/GermanSpeechRecognition
dev_path = '/data/s3757994/dev.tsv'
train_path = '/data/s3757994/train.tsv'
test_path = '/data/s3757994/test.tsv'
root = '/data/s3757994/clips_wav/
n_jobs = -2
n_filters = 40
win_size = 0.025/3
norm_x = False

# def dividedataset(root):
#     files = os.listdir(root)
#     numfiles = len(files)
#     train = files[:int(0.7*numfiles)]
#     trainlabels = []
#     dev = files[int(0.7*numfiles):int(0.9 *numfiles)]
#     devlabels = []
#     test = files[int(0.9*numfiles):]
#     testlabels = []
#     validated = open('/data/s3757994/validated.tsv',"r")
#     reader = csv.reader(validated, delimiter="\t")
#     for row in reader:
#         if row[1]+".wav" in train:
#





def traverse(root,path,search_fix='.wav',return_label=False):
    files = os.listdir(root)
    numfiles = len(files)
    print(numfiles)
    print(files[:5])
    if path == "train":
        set = files[:int(0.7*numfiles)]
    elif path == "dev":
        set = files[int(0.7*numfiles):int(0.9 *numfiles)]
    else:
        set = files[int(0.9*numfiles):]
    f_list = []
    with open('/data/s3757994/validated.tsv') as tsvfile:
      reader = csv.reader(tsvfile, delimiter='\t')
      first = True
      counter = 0
      for row in reader:
          if first:
              first = False
          else:
              if (row[1] + ".wav") in set:
              # print(row[1])
                  if return_label:
                      f_list.append(row[2])
                  else:
                      f_list.append(root + row[1]+".wav")
          # counter += 1
    return f_list

def flac2wav(f_path):
    flac_audio = AudioSegment.from_file(f_path, "flac")
    flac_audio.export(f_path[:-5]+'.wav', format="wav")

def wav2logfbank(f_path):
    (rate,sig) = wav.read(f_path)
    fbank_feat = logfbank(sig,rate,winlen=win_size,nfilt=n_filters)
    np.save(f_path[:-3]+'fb'+str(n_filters),fbank_feat)

def norm(f_path,mean,std):
    np.save(f_path,(np.load(f_path)-mean)/std)


print('----------Processing Datasets----------')
print('Training sets :',train_path)
print('Validation sets :',dev_path)
print('Testing sets :',test_path)

# # print('Training',flush=True)
tr_file_list = traverse(root,"train")
# print(tr_file_list[0])
# # results = Parallel(n_jobs=n_jobs,backend="threading")(delayed(flac2wav)(i) for i in tqdm(tr_file_list))
#
# print('Validation')
dev_file_list = traverse(root,"dev")
# print(dev_file_list[0])
# results = Parallel(n_jobs=n_jobs,backend="threading")(delayed(flac2wav)(i) for i in tqdm(dev_file_list))
#
# # print('Testing',flush=True)
tt_file_list = traverse(root,"test")
# results = Parallel(n_jobs=n_jobs,backend="threading")(delayed(flac2wav)(i) for i in tqdm(tt_file_list))



# # wav 2 log-mel fbank
print('---------------------------------------')
print('Processing wav2logfbank...')

# print('Training',flush=True)
results = Parallel(n_jobs=n_jobs,backend="threading")(delayed(wav2logfbank)(i[:-3]+'wav') for i in tqdm(tr_file_list))

print('Validation')
results = Parallel(n_jobs=n_jobs,backend="threading")(delayed(wav2logfbank)(i[:-3]+'wav') for i in tqdm(dev_file_list))

# print('Testing',flush=True)
results = Parallel(n_jobs=n_jobs,backend="threading")(delayed(wav2logfbank)(i[:-3]+'wav') for i in tqdm(tt_file_list))



# # log-mel fbank 2 feature
print('---------------------------------------')
print('Preparing Training Dataset...')

tr_file_list = traverse(root,"train",search_fix='.fb'+str(n_filters))
tr_text = traverse(root,"train",return_label=True)

X = []
for f in tr_file_list:
    X.append(np.load(f[:-3] +"fb40.npy"))

# Normalize X
if norm_x:
    mean_x = np.mean(np.concatenate(X,axis=0),axis=0)
    std_x = np.std(np.concatenate(X,axis=0),axis=0)

    results = Parallel(n_jobs=n_jobs,backend="threading")(delayed(norm)(i,mean_x,std_x) for i in tqdm(tr_file_list))


# Sort data by signal length (long to short)
audio_len = [len(x) for x in X]

tr_file_list = [tr_file_list[idx] for idx in reversed(np.argsort(audio_len))]
tr_text = [tr_text[idx] for idx in reversed(np.argsort(audio_len))]
#
# # Create char mapping
char_map = {}
char_map['<sos>'] = 0
char_map['<eos>'] = 1
char_map['/'] = 2
char_map['…'] = 3
char_map['@'] = 4
char_map['ş'] = 5
char_map['ó'] = 6
char_map['ú'] = 7
char_map['à'] = 8
char_map['è'] = 9
char_map['ì'] = 10
char_map['ò'] = 11
char_map['ù'] = 12

char_idx = 12

# map char to index
for text in tr_text:
    for char in text:
        if char not in char_map:
            char_map[char] = char_idx
            char_idx +=1

for k,v in char_map.items():
    print(k)

# Reverse mapping
rev_char_map = {v:k for k,v in char_map.items()}

# Save mapping
with open(root+'idx2chap.csv','w') as f:
    f.write('idx,char\n')
    for i in range(len(rev_char_map)):
        f.write(str(i)+','+rev_char_map[i]+'\n')

# text to index sequence
tmp_list = []
for text in tr_text:
    tmp = []
    for char in text:
        tmp.append(char_map[char])
    tmp_list.append(tmp)
tr_text = tmp_list
del tmp_list

# write dataset
file_name = 'train.csv'

print('Writing dataset to '+root+file_name+'...',flush=True)

with open(root+file_name,'w') as f:
    f.write('idx,input,label\n')
    for i in range(len(tr_file_list)):
        f.write(str(i)+',')
        f.write(tr_file_list[i]+',')
        for char in tr_text[i]:
            f.write(' '+str(char))
        f.write('\n')

print()
print('Preparing Validation Dataset...',flush=True)

dev_file_list = traverse(root,"dev",search_fix='.fb'+str(n_filters))
print(dev_file_list[0])
dev_text = traverse(root,"dev",return_label=True)



X = []
for f in dev_file_list:
    X.append(np.load(f[:-3] +"fb40.npy"))

print("yeah joe joe")
# Normalize X
if norm_x:
    results = Parallel(n_jobs=n_jobs,backend="threading")(delayed(norm)(i,mean_x,std_x) for i in tqdm(dev_file_list))


# Sort data by signal length (long to short)
audio_len = [len(x) for x in X]

dev_file_list = [dev_file_list[idx] for idx in reversed(np.argsort(audio_len))]
dev_text = [dev_text[idx] for idx in reversed(np.argsort(audio_len))]

# text to index sequence
tmp_list = []
for text in dev_text:
    tmp = []
    for char in text:
        try:
            tmp.append(char_map[char])
        except:
            print(char)
    tmp_list.append(tmp)
dev_text = tmp_list
del tmp_list



# write dataset
file_name = 'dev.csv'

print('Writing dataset to '+root+file_name+'...',flush=True)

with open(root+file_name,'w') as f:
    f.write('idx,input,label\n')
    for i in range(len(dev_file_list)):
        f.write(str(i)+',')
        f.write(dev_file_list[i]+',')
        for char in dev_text[i]:
            f.write(' '+str(char))
        f.write('\n')

print()
print('Preparing Testing Dataset...',flush=True)

test_file_list = traverse(root,"test",search_fix='.fb'+str(n_filters))
tt_text = traverse(root,"test",return_label=True)

X = []
for f in test_file_list:
    X.append(np.load(f[:-3] +"fb40.npy"))

# Normalize X
if norm_x:
    results = Parallel(n_jobs=n_jobs,backend="threading")(delayed(norm)(i,mean_x,std_x) for i in tqdm(test_file_list))


# Sort data by signal length (long to short)
audio_len = [len(x) for x in X]

test_file_list = [test_file_list[idx] for idx in reversed(np.argsort(audio_len))]
tt_text = [tt_text[idx] for idx in reversed(np.argsort(audio_len))]

# text to index sequence
tmp_list = []
for text in tt_text:
    tmp = []
    for char in text:
        try:
            tmp.append(char_map[char])
        except:
            print(char)
    tmp_list.append(tmp)
tt_text = tmp_list
del tmp_list

# write dataset
file_name = 'test.csv'

print('Writing dataset to '+root+file_name+'...',flush=True)

with open(root+file_name,'w') as f:
    f.write('idx,input,label\n')
    for i in range(len(test_file_list)):
        f.write(str(i)+',')
        f.write(test_file_list[i]+',')
        for char in tt_text[i]:
            f.write(' '+str(char))
        f.write('\n')
