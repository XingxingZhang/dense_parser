
curdir=`pwd`
codedir=/afs/inf.ed.ac.uk/group/project/img2txt/dep_parser/dense_release

model=$curdir/model_0.001.tune.t7
validout=$curdir/valid
testout=$curdir/test
log=$curdir/log.txt

cd $codedir

CUDA_VISIBLE_DEVICES=2 th mst_postprocess.lua \
    --mstalg Eisner \
    --modelPath $model \
    --validout $validout \
    --testout $testout | tee $log

cd $curdir

