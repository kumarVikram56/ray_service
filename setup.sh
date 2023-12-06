#!/bin/bash

# write all commands need to setup this project

cd vectorDbSrv
mkdir library
cd library
git clone https://github.com/HowieMa/DeepSORT_YOLOv5_Pytorch.git
cd ..
pip3 install -r requirements.txt
uvicorn main:app --reload --port 3000