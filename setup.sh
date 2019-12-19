#!/bin/bash
mkdir -p data
mkdir -p data/input
mkdir -p data/output
mkdir -p features
mkdir -p folds
mkdir -p logs
mkdir -p pickle

touch configs/common/notify.yml

kaggle competitions download -c ashrae-energy-prediction -p ./data/input
cd ./data/input
unzip ashrae-energy-prediction.zip