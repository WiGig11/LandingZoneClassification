#!/bin/bash


if ! command -v python &> /dev/null; then
    echo "Python could not be found"
    exit 1
fi


python -c "
import torch
if torch.cuda.is_available():
    print('GPU is available')
    print('Current GPU device:', torch.cuda.current_device())
else:
    print('No GPU available')
"
