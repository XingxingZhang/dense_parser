
# ID=`./gpu_lock.py --id-to-hog 0`
ID=3
echo $ID
if [ $ID -eq -1 ]; then
    echo "this gpu is not free"
    exit
fi
# ./gpu_lock.py

codedir=/afs/inf.ed.ac.uk/group/project/img2txt/dep_parser/dense_release
curdir=`pwd`
lr=0.001
label=.std.ft0.dp0.4.r0.1.bs20
model=$curdir/model_$lr$label.t7
log=$curdir/log_$lr$label.txt

: '
train=/disk/scratch/XingxingZhang/dep_parse/dataset/german/german_gold_train.conll
valid=/disk/scratch/XingxingZhang/dep_parse/dataset/german/german_gold_dev.conll
test=/disk/scratch/XingxingZhang/dep_parse/dataset/german/german_gold_test.conll
'
train=/disk/scratch/s1270921/dep_parse/data_conll/german/german_gold_train.conll
valid=/disk/scratch/s1270921/dep_parse/data_conll/german/german_gold_dev.conll
test=/disk/scratch/s1270921/dep_parse/data_conll/german/german_gold_test.conll



cd $codedir

CUDA_VISIBLE_DEVICES=$ID th train.lua --useGPU \
    --model SelectNetPos \
    --seqLen 112 \
    --maxTrainLen 110 \
    --freqCut 1 \
    --nhid 300 \
    --nin 300 \
    --nlayers 2 \
    --dropout 0.4 \
    --recDropout 0.1 \
    --lr $lr \
    --train $train \
    --valid $valid \
    --test $test \
    --optimMethod Adam \
    --save $model \
    --batchSize 20 \
    --validBatchSize 20 \
    --maxEpoch 15 \
    --npin 40 \
    --evalType conllx \
    | tee $log

cd $curdir

# ./gpu_lock.py --free $ID
# ./gpu_lock.py

