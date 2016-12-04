
ID=2
echo $ID
if [ $ID -eq -1 ]; then
    echo "this gpu is not free"
    exit
fi

codedir=/afs/inf.ed.ac.uk/group/project/img2txt/dep_parser/dense_release
curdir=`pwd`
lr=0.001
label=.tune
model=$curdir/model_$lr$label.t7
log=$curdir/log_$lr$label.txt

load=$curdir/model_0.001.std.ft0.lpos.dp.t7

cd $codedir

CUDA_VISIBLE_DEVICES=$ID th post_train.lua \
    --load $load \
    --save $model \
    --lr $lr \
    --maxEpoch 10 \
    --optimMethod SGD \
    | tee $log

cd $curdir


