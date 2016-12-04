
curdir=`pwd`
codedir=/afs/inf.ed.ac.uk/group/project/img2txt/dep_parser/dense_release

model=/disk/scratch/s1270921/dep_parse/experiments/English/release/model_0.001.tune.t7
validout=$curdir/valid
testout=$curdir/test
log=$curdir/mst_log.txt

cd $codedir

CUDA_VISIBLE_DEVICES=0 th mst_postprocess.lua \
    --mstalg Eisner \
    --modelPath $model \
    --validout $validout \
    --testout $testout | tee $log

cd $curdir

