#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB

module load FFmpeg
/home/s3757994/GermanSpeechRecognition/toflac.py
/home/s3757994/GermanSpeechRecognition/util/preprocess_mozilla.py
