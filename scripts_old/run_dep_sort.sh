
source ~/.profile 

train=../../dataset/penn_wsj.dep.train
valid=../../dataset/penn_wsj.dep.valid
test=../../dataset/penn_wsj.dep.test
dataset=../../dataset/penn_wsj.dep.h5
th deptree2hdf5.lua --train $train --valid $valid --test $test --dataset $dataset --freq 5 --keepFreq --maxLen 150 --ignoreCase --sort -1 --batchSize 64

