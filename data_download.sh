#!/bin/bash
curl -L -o ./archive.zip \
https://www.kaggle.com/api/v1/datasets/download/hojjatk/mnist-dataset

# unzip the downloaded file into data folder
unzip archive.zip -d ./data

# remove the downloaded file
rm archive.zip