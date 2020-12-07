#createressum_v2: Create Result Summary for all the methods in a single file
#Takes one argument: one of {GNN, VEGNN, CEGNN, VECEGNN, SEGNN, BotGNN}

#!/bin/bash


#source directory names:
res1="Result_BotGNN1_Czech_AB"
res2="Result_BotGNN2_Czech_AB"
res3="Result_BotGNN3_Czech_AB"
res4="Result_BotGNN4_Czech_AB"
res5="Result_BotGNN5_Czech_AB"


#Result summary file:
resfile="./Results/BotGNN_AB.csv"

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

