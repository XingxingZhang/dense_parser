
source ~/.profile 

train=../../dataset/depparse/train.autopos.s1
valid=../../dataset/depparse/valid.autopos
test=../../dataset/depparse/test.autopos
dataset=../../dataset/penn_wsj.conllx.s1.h5
th conllx2hdf5.lua --train $train --valid $valid --test $test --dataset $dataset --freq 1 --keepFreq --maxLen 150 --sort 0 --batchSize 64

