#!/bin/bash

# Version: 2

# Written by: Tirtharaj Dash, Ph.D. Student, BITS Pilani, Goa Campus, India
# Date: During December 2019
# E-mail: tirtharaj@goa.bits-pilani.ac.in

# Work: BotGNN (Graph Neural Network based on ILP Bottom Clause)
	# A bottom clause in ILP can be written as a set of relation and entities.
	# We treat each relation as a relation(R) node and an entity as an entity(E) node
	# This forms a graph (possibly, a bi-partite) graph of R-E nodes
	# This work is intended towards making the the idea of incorporating domain-knowledge generic for any ILP problem. 
# Purpose: bash script to run graph classification methods on NCI data (73 problems). 
# Learning Type: Classification (Binary)

# Result storage mapping: [ Methods: BotGNN | ID: 1--5 (networks1--5.py) ]


#path settings
prefixdir="/home/dell5810/tdash/prepareBotGraph/processedBOTDS"
trntstsplitdir="/home/dell5810/tdash/DataForVEGNN/TrainTestSplit"


#create the directory where the Results will be stored
resultdir="Result_BotGNN5_Czech"

if [ ! -d $resultdir ]
then 
	mkdir $resultdir
fi

#for each dataset in the list
for dataset in `cat datasets`
do
	echo "Working on: $dataset"

	#copy the input: train and test files to run dir
	rm -rf ./data/BOTDS/*
	mkdir ./data/BOTDS/raw
	cp $prefixdir/$dataset/BOTDS_*.txt ./data/BOTDS/raw/.
	
	#copy the train_test split info
	cp $trntstsplitdir/$dataset/*_split ./data/BOTDS/.

	#run the python program
	python main.py --dataset BOTDS
	
	#store the results
	if [ -d ./$resultdir/$dataset ]
	then
		rm -rf ./$resultdir/$dataset
	fi
	mkdir ./$resultdir/$dataset

	#mv ./data/DS/* $resultdir/$dataset/.
	mv score.txt $resultdir/$dataset/.
	mv latest.pth $resultdir/$dataset/.
done

