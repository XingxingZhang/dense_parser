
# ID=`./gpu_lock.py --id-to-hog 0`
ID=2
echo $ID
if [ $ID -eq -1 ]; then
    echo "this gpu is not free"
    exit
fi
# ./gpu_lock.py

codedir=/afs/inf.ed.ac.uk/group/project/img2txt/dep_parser/dense_release

curdir=`pwd`
lr=0.001
label=.std.ft0.lpos.dp
model=$curdir/model_$lr$label.t7
log=$curdir/log_$lr$label.txt

train=/disk/scratch/s1270921/dep_parse/dataset/ctb/train.gold.conll
valid=/disk/scratch/s1270921/dep_parse/dataset/ctb/dev.gold.conll
test=/disk/scratch/s1270921/dep_parse/dataset/ctb/test.gold.conll

wembed=/disk/scratch/s1270921/dep_parse/dataset/ctb/vectors.ctb.t7

cd $codedir

CUDA_VISIBLE_DEVICES=$ID th train.lua --useGPU \
    --model SelectNetPos \
    --seqLen 142 \
    --maxTrainLen 140 \
    --freqCut 1 \
    --nhid 300 \
    --nin 300 \
    --nlayers 2 \
    --dropout 0.35 \
    --recDropout 0.05 \
    --lr $lr \
    --train $train \
    --valid $valid \
    --test $test \
    --optimMethod Adam \
    --save $model \
    --batchSize 20 \
    --validBatchSize 20 \
    --maxEpoch 15 \
    --wordEmbedding $wembed \
    --embedOption fineTune \
    --fineTuneFactor 0 \
    --npin 50 \
    | tee $log


cd $curdir

# ./gpu_lock.py --free $ID
# ./gpu_lock.py

