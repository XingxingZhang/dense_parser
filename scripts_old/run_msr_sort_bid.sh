
dataset=/disk/scratch/XingxingZhang/treelstm/dataset/msr/msr.dep.100.bid.h5
th sort_large_hdf5_bid.lua --dataset $dataset --sort -1 --batchSize 64 --bidirectional

