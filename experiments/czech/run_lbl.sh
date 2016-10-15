
codedir=/afs/inf.ed.ac.uk/group/project/img2txt/dep_parser/dense_release
curdir=`pwd`

dataset=$curdir/label_train.h5
model=$curdir/lbl_lassifier.t7

inTrain=/disk/scratch1/XingxingZhang/dep_parse/dataset/czech/czech_gold_train.conll
inValid=/disk/scratch1/XingxingZhang/dep_parse/dataset/czech/czech_gold_dev.conll
inTest=/disk/scratch1/XingxingZhang/dep_parse/dataset/czech/czech_gold_test.conll

log=$curdir/lbl_log.txt

cd $codedir

CUDA_VISIBLE_DEVICES=3 th train_labeled.lua --mode train \
	--useGPU \
	--snhids "1880,800,800,82" \
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
	--language Other \
	--save $model | tee $log

cd $curdir

