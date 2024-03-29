from pydub import AudioSegment
from os import listdir
from os.path import isfile, join
import csv

import_folder = "/data/s3757994/clips"
export_folder = "/data/s3757994/clips_wav"
files = [f for f in listdir(import_folder) if isfile(join(import_folder, f))]

# f_list = []
# with open('/data/s3757994/de/dev.tsv') as tsvfile:
#   reader = csv.reader(tsvfile, delimiter='\t')
#   first = True
#   counter = 0
#   for row in reader:
#       if first:
#           first = False
#       else:
#           f_list.append(row[1]+".mp3")
#
#       counter += 1


for file in files[90000:]:
    # file = file[:-3]

    try:
        song = AudioSegment.from_mp3(import_folder + "/" +file)
        song.export(export_folder+"/"+file[:-4]+".wav",format = "wav")

        # print(export_folder+"/"+file+"wav")
    except:
        print("continue")
        print(file)
        continue
