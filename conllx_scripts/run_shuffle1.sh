
seed=1
infile=/afs/inf.ed.ac.uk/group/project/img2txt/xTreeLSTM/xtreelstm/dataset/depparse/train.autopos
outfile=/afs/inf.ed.ac.uk/group/project/img2txt/xTreeLSTM/xtreelstm/dataset/depparse/train.autopos.s$seed
th shuffle.lua --inFile $infile --outFile $outfile --seed $seed

