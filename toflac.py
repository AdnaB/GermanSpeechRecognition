from pydub import AudioSegment
from os import listdir
from os.path import isfile, join

import_folder = "./clips"
export_folder = "./clips_flac"
files = [f for f in listdir(import_folder) if isfile(join(import_folder, f))]

for file in files:
    file = file[:-3]
    song = AudioSegment.from_mp3(import_folder+"/"+file+"mp3")
    song.export(export_folder+"/"+file+".flac",format = "flac")
    print(export_folder+"/"+file+"flac")
