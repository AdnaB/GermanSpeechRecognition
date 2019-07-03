#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --partition=cpu
#SBATCH --mem=32GB
module load Python/3.6.4-foss-2018a
module load FFmpeg
python3 /data/s3757994/GermanSpeechRecognition/toflac.py
python3 /data/s3757994/GermanSpeechRecognition/util/preprocess_mozilla.py
