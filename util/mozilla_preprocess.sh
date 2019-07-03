if [ "$#" -ne 1 ]; then
    echo "Usage : ./mozilla_preprocess.sh <Mozillafolder>"
fi

python3 toflac.py

python3 preprocess_mozilla.py
