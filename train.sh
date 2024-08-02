#!/bin/bash

#python train.py --mode 'train' --batch_size 8 --num_worker 2 --epoches 5
#python train.py --mode 'train' --logfilename 'test' --model_savepath '/chz/models' --epoches 5 --tensorboardname 'test'
python train.py --mode 'test' --logfilename 'test' --model_savepath '/chz/models' --epoches 5 --tensorboardname 'test'