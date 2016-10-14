
vocab=/disk/scratch1/xingxing.zhang/xTreeRNN/dataset/msr/msr.dep.100.sort20.vocab.t7
test=/afs/inf.ed.ac.uk/group/project/img2txt/deptree_rnnlm/data/msr_sent_compl/question.dep
testdataset=/disk/scratch1/xingxing.zhang/xTreeRNN/dataset/msr/msr.dep.100.question.h5
th testsetdeptree2hdf5.lua --vocab $vocab --test $test --testdataset $testdataset
