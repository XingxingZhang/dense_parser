
source ~/.profile 

train=../../dataset/penn_wsj.train
valid=../../dataset/penn_wsj.valid
test=../../dataset/penn_wsj.test
dataset=../../dataset/penn_wsj.h5
# th words2hdf5.lua --train $train --valid $valid --test $test --dataset $dataset --freq 0 --ignorecase --keepfreq --maxlen 1000
th words2hdf5.lua --train $train --valid $valid --test $test --dataset $dataset --freq 0 --keepfreq --maxlen 100

