#! /bin/bash

conda env create -f requirements.yml
conda activate python311
pip install opencv-python
pip install pandas
pip install mediapipe