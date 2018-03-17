#!/usr/bin/env sh
echo "Create train.txt..."
rm -rf train.txt
j=0
for i in 1 y 3 4 5 6 7 8 9 a b c d e
do
find trainset -name $i*.png | cut -d '/' -f2 | sed "s/$/ $j/">>train.txt
j=$(($j+1))
done

echo "Create test.txt..."
rm -rf test.txt
j=0
for i in 1 y 3 4 5 6 7 8 9 a b c d e
do
find testset -name $i*.png | cut -d '/' -f2 | sed "s/$/ $j/">>test.txt
j=$(($j+1))
done

