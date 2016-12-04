
codedir=/afs/inf.ed.ac.uk/group/project/img2txt/dep_parser/dense_release
curdir=`pwd`

model=$curdir/model_0.001.tune.t7

outTrain=$curdir/label_train.h5

inTrain=/afs/inf.ed.ac.uk/group/project/img2txt/dep_parser/dataset/train.autopos
inValid=/afs/inf.ed.ac.uk/group/project/img2txt/dep_parser/dataset/valid.autopos
inTest=/afs/inf.ed.ac.uk/group/project/img2txt/dep_parser/dataset/test.autopos

outValid=$curdir/valid.dep
outTest=$curdir/test.dep

log=$curdir/gen-log.txt

cd $codedir

CUDA_VISIBLE_DEVICES=3 th train_labeled.lua --mode generate \
	--modelPath $model \
	--outTrainDataPath $outTrain \
	--inTrain $inTrain \
	--inValid $inValid \
	--inTest $inTest \
	--outValid $outValid \
	--outTest $outTest \
	--language English | tee $log

cd $curdir

