from pydub import AudioSegment
from os import listdir
from os.path import isfile, join
import csv

import_folder = "./de/clips"
export_folder = "./de/clips_flac"
files = [f for f in listdir(import_folder) if isfile(join(import_folder, f))]

f_list = []
with open('/home/sanne/Documents/RUG/DeepLearning/GermanSpeechRecognition/de/dev.tsv') as tsvfile:
  reader = csv.reader(tsvfile, delimiter='\t')
  first = True
  counter = 0
  for row in reader:
      if first or counter > 25:
          first = False
      else:
          f_list.append(row[1]+".mp3")

      counter += 1


for file in f_list:
    # file = file[:-3]
    song = AudioSegment.from_mp3(import_folder + "/" +file)
    song.export(export_folder+"/"+file[:-4]+".flac",format = "flac")
    try:


        print(export_folder+"/"+file+"flac")
    except:
        print("continue")
        print(file)
        continue
