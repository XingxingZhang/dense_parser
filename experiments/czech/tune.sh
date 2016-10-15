
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
label=.tune.ori
model=$curdir/model_$lr$label.t7
log=$curdir/log_$lr$label.txt

load=/disk/scratch1/XingxingZhang/dep_parse/experiments/release_test/czech/model_0.001.dp0.35.r0.1.bs20.ori.t7

cd $codedir

CUDA_VISIBLE_DEVICES=$ID th post_train.lua \
    --load $load \
    --save $model \
    --lr $lr \
    --maxEpoch 10 \
    --optimMethod SGD \
    | tee $log

cd $curdir

# ./gpu_lock.py --free $ID
# ./gpu_lock.py

