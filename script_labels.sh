#!/bin/bash

#sostituisce la prima lettera di ogni riga con 0, (serve a risolvere il problema delle classi di LabelImg)
cd "data/labels"
#echo "$PWD"
ls

for FILE in *; 
    do 
        #echo $FILE;
        sed 's/^\s*./0/g' $FILE > test
        mv test $FILE
        
done

