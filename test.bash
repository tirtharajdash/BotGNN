#!/bin/bash

# Written by: Tirtharaj Dash, Ph.D. Student, BITS Pilani, Goa Campus, India
# Date: During Jan-April 2019
# E-mail: tirtharaj@goa.bits-pilani.ac.in

# Purpose: bash script to test saved models from the work of graph classification methods on NCI data (73 problems). 
# Learning Type: Classification (Binary)

# Result storage mapping: [ Methods: BotGNN | ID: 1 (networks1.py), 2 (networks2.py), 3 (networks3.py), 4 (networks4.py), 4_1 (networks4_1.py) ]


#path settings
prefixdir="/home/dell5810/tdash/prepareBotGraph/processedBOTDS"
trntstsplitdir="/home/dell5810/tdash/DataForVEGNN/TrainTestSplit"

#pass the path of the directory, where Results are stored.
resultdir="Result_BotGNN4_Czech"

for dataset in `cat datasets | head -10`
do
	echo "Working on: $dataset"

    #copy the input: train and test files to run dir
    rm -rf ./data/BOTDS/*
    mkdir ./data/BOTDS/raw
    cp $prefixdir/$dataset/BOTDS_*.txt ./data/BOTDS/raw/.

    #copy the train_test split info
    cp $trntstsplitdir/$dataset/*_split ./data/BOTDS/.

	#copy saved model from results path and score file
	ln -s $resultdir/$dataset/* .
	cat score.txt | gawk '{print $3}'

	#test & print the saved model
	#python modelsummary.py #this is already included in the evalsavedmodel.py
	python evalsavedmodel.py #to print the model summary here: uncomment the line "print(model)" in the evalsavedmodel.py script

	#save the individual instance predictions
	mkdir -p BotGNN_preds/$dataset
	mv preds.txt BotGNN_preds/$dataset/test_preds
	rm score.txt latest.pth
done

