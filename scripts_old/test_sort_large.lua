
require 'hdf5'
require 'torch'
include '../utils/shortcut.lua'
require 'deptreeutils'

require 'sorthdf5'

local opts = {}
opts.dataset = '/disk/scratch1/xingxing.zhang/xTreeRNN/dataset/msr/msr.dep.100.h5'
opts.sort = 20
opts.batchSize = 64
local h5sortFile = '/disk/scratch1/xingxing.zhang/xTreeRNN/dataset/msr/msr.dep.100.sort20.h5'
local h5sorter = SortHDF5(opts.dataset, h5sortFile)
h5sorter:sortHDF5(opts.sort, opts.batchSize)