
codedir=/afs/inf.ed.ac.uk/group/project/img2txt/dep_parser/dense_release
curdir=`pwd`

model=/disk/scratch/s1270921/dep_parse/experiments/pre-trained-models/german/model_0.001.tune.t7
classifier=/disk/scratch/s1270921/dep_parse/experiments/pre-trained-models/german/lbl_classifier.t7
input=/disk/scratch/s1270921/dep_parse/data_conll/german/german_gold_test.conll
output=$curdir/german.conllx

cd $codedir

CUDA_VISIBLE_DEVICES=3 th dense_parser.lua --modelPath $model --classifierPath $classifier \
	--input $input --output $output --gold $input --mstalg ChuLiuEdmonds

cd $curdir

