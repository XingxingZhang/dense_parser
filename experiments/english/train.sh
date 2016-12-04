
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
label=.std.ft0.300.dp0.35.r0.1
model=$curdir/model_$lr$label.t7
log=$curdir/log_$lr$label.txt

train=/afs/inf.ed.ac.uk/group/project/img2txt/dep_parser/dataset/train.autopos
valid=/afs/inf.ed.ac.uk/group/project/img2txt/dep_parser/dataset/valid.autopos
test=/afs/inf.ed.ac.uk/group/project/img2txt/dep_parser/dataset/test.autopos

wembed=/afs/inf.ed.ac.uk/group/project/img2txt/dep_parser/dataset/glove.840B.300d.penn.t7

cd $codedir

CUDA_VISIBLE_DEVICES=$ID th train.lua --useGPU \
    --model SelectNetPos \
    --seqLen 120 \
    --maxTrainLen 100 \
    --freqCut 1 \
    --nhid 300 \
    --nin 300 \
    --nlayers 2 \
    --dropout 0.35 \
    --recDropout 0.1 \
    --lr $lr \
    --train $train \
    --valid $valid \
    --test $test \
    --optimMethod Adam \
    --save $model \
    --batchSize 32 \
    --validBatchSize 32 \
    --maxEpoch 15 \
    --wordEmbedding $wembed \
    --embedOption fineTune \
    --fineTuneFactor 0 \
    --npin 30 \
    | tee $log

cd $curdir

# ./gpu_lock.py --free $ID
# ./gpu_lock.py

