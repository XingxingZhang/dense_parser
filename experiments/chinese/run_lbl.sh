
codedir=/afs/inf.ed.ac.uk/group/project/img2txt/dep_parser/dense_release
curdir=`pwd`

dataset=$curdir/label_train.h5
model=$curdir/lbl_lassifier.t7

inTrain=/disk/scratch/s1270921/dep_parse/dataset/ctb/train.gold.conll
inValid=/disk/scratch/s1270921/dep_parse/dataset/ctb/dev.gold.conll
inTest=/disk/scratch/s1270921/dep_parse/dataset/ctb/test.gold.conll

log=$curdir/lbl_log.txt

cd $codedir

CUDA_VISIBLE_DEVICES=2 th train_labeled.lua --mode train \
	--useGPU \
	--snhids "1900,800,800,12" \
	--activ relu \
	--lr 0.01 \
	--optimMethod AdaGrad \
	--dropout 0.5 \
	--inDropout 0.05 \
	--batchSize 256 \
	--maxEpoch 20 \
	--ftype "|x|xe|xpe|" \
	--dataset $dataset \
	--inTrain $inTrain \
	--inValid $inValid \
	--inTest $inTest \
	--save $model | tee $log

cd $curdir


