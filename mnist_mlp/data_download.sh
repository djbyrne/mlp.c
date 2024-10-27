#!/bin/bash
curl -L -o ./archive.zip \
https://www.kaggle.com/api/v1/datasets/download/hojjatk/mnist-dataset

# unzip the downloaded file
unzip archive.zip

# remove the downloaded file
rm archive.zip