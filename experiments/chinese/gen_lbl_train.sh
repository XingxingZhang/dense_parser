
codedir=/afs/inf.ed.ac.uk/group/project/img2txt/dep_parser/dense_release

curdir=`pwd`

model=$curdir/model_0.001.tune.t7

outTrain=$curdir/label_train.h5

inTrain=/disk/scratch/s1270921/dep_parse/dataset/ctb/train.gold.conll
inValid=/disk/scratch/s1270921/dep_parse/dataset/ctb/dev.gold.conll
inTest=/disk/scratch/s1270921/dep_parse/dataset/ctb/test.gold.conll


outValid=$curdir/valid.dep
outTest=$curdir/test.dep

log=$curdir/gen-log.txt

cd $codedir

CUDA_VISIBLE_DEVICES=2 th train_labeled.lua --mode generate \
	--modelPath $model \
	--outTrainDataPath $outTrain \
	--inTrain $inTrain \
	--inValid $inValid \
	--inTest $inTest \
	--outValid $outValid \
	--outTest $outTest \
	--language Chinese | tee $log

cd $curdir

