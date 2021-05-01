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
# Purpose: bash script to run graph classification methods on Mutagenesis data (10 splits). 
# Learning Type: Classification (Binary)

# Result storage mapping: [ Methods: BotGNN | ID: 1--5 (networks1--5.py) ]


#path settings
prefixdir="/home/dell5810/tdash/prepareBotGraph/Amine/processed_type"
trntstsplitdir="/home/dell5810/tdash/prepareBotGraph/Amine/amine_splits"


#create the directory where the Results will be stored
resultdir="Result_BotGNN4_Amine_type"

if [ ! -d $resultdir ]
then 
	mkdir $resultdir
fi

#for each dataset in the list
for split in {1..10}
do
	echo "Working on split: $split"

	#copy the input: train and test files to run dir
	rm -rf ./data/BOTDS/*
	mkdir ./data/BOTDS/raw
	cp $prefixdir/BOTDS_*.txt ./data/BOTDS/raw/.
	
	#copy the train_test split info
	cp $trntstsplitdir/s$split/*_split ./data/BOTDS/.

	#run the python program
	python main_amine.py --dataset BOTDS
	
	#store the results
	if [ -d ./$resultdir/s$split ]
	then
		rm -rf ./$resultdir/s$split
	fi
	mkdir ./$resultdir/s$split

	#mv ./data/DS/* $resultdir/s$split/.
	mv score.txt $resultdir/s$split/.
	mv latest.pth $resultdir/s$split/.
done
