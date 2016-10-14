
source ~/.profile 

train=../../dataset/depparse/train.autopos
valid=../../dataset/depparse/valid.autopos
test=../../dataset/depparse/test.autopos
dataset=../../dataset/penn_wsj.conllx.bid.h5
th conllx2hdf5.lua --train $train --valid $valid --test $test --dataset $dataset --freq 1 --keepFreq --maxLen 150 --sort -1 --batchSize 64 --bidirectional

