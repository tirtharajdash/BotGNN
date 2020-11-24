#createressum_v2: Create Result Summary for all the methods in a single file
#Takes one argument: one of {GNN, VEGNN, CEGNN, VECEGNN, SEGNN, BotGNN}

#!/bin/bash

if [ $# -eq 0 ]
then
	echo "No method is passed as argument; see help with: bash $0 --help"
	exit 1
fi

if [ "$1" == "--help" ]
then
	echo "Help: Enter a filename as argument to the program."
	echo "Example: bash $0 VEGNN"
	exit 1
fi

#source directory names:
res1="Result_$1""1_Czech"
res2="Result_$1""2_Czech"
res3="Result_$1""3_Czech"
res4="Result_$1""4_Czech"
res5="Result_$1""5_Czech"


#Result summary file:
resfile="./Results/$1.csv"

#start writing out to the file
echo -e "DATASET,VARIANT1,VARIANT2,VARIANT3,VARIANT4,VARIANT5" > $resfile
for dataset in `cat datasets`
do
	a=`cat $res1/$dataset/score.txt | gawk '{print $3}'`
	b=`cat $res2/$dataset/score.txt | gawk '{print $3}'`
	c=`cat $res3/$dataset/score.txt | gawk '{print $3}'`
	d=`cat $res4/$dataset/score.txt | gawk '{print $3}'`
	e=`cat $res5/$dataset/score.txt | gawk '{print $3}'`
	echo -e "$dataset,$a,$b,$c,$d,$e" >> $resfile
done

sed -i 's/gi50_screen_//g' $resfile 

