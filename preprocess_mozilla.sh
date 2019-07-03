#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
module load Python/3.6.4-foss-2018a
module load FFmpeg
/data/s3757994/GermanSpeechRecognition/toflac.py
/data/s3757994/GermanSpeechRecognition/util/preprocess_mozilla.py