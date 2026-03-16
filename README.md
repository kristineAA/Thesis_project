# activate venv
source tb_venv/bin/activate
# run script
python PredictionS2_tensorflow.py
# start tensorboard
python -m tensorboard.main --logdir "$(pwd)/runs" --load_fast=false
# open browser
http://localhost:6006/