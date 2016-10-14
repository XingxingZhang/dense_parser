
require 'hdf5'
require 'torch'
include '../utils/shortcut.lua'
require 'deptreeutils'
require 'sorthdf5'

local function getOpts()
  local cmd = torch.CmdLine()
  cmd:text('== convert dependency trees to hdf5 format ==')
  cmd:text()
  cmd:text('Options:')
  cmd:option('--dataset', '', 'the resulting dataset (.h5)')
  cmd:option('--sort', 0, '0: no sorting of the training data; -1: sort training data by their length; k (k > 0): sort the consecutive k batches by their length')
  cmd:option('--batchSize', 64, 'batch size when --sort > 0 or --sort == -1')
  
  return cmd:parse(arg)
end

local function main()
  local opts = getOpts()
  local dataPrefix = opts.dataset:sub(1, -4)
  local vocabPath = dataPrefix .. '.vocab.t7'
  if opts.sort ~= 0 then
    assert(opts.sort == -1 or opts.sort > 0, 'valid values [0, -1, > 0]')
    print '========begin to sort dataset========'
    local h5sorter = nil
    local mid = opts.sort == -1 and 'sort' or string.format('sort%d', opts.sort)
    local h5sortFile = opts.dataset:sub(1, -4) .. string.format('.%s.h5', mid)
    local h5sortVocabFile = opts.dataset:sub(1, -4) .. string.format('.%s.vocab.t7', mid)
    local cmd = string.format('cp %s %s', vocabPath, h5sortVocabFile)
    print(cmd)
    os.execute(cmd)
    h5sorter = SortHDF5(opts.dataset, h5sortFile)
    h5sorter:sortHDF5(opts.sort, opts.batchSize)
    printf('save dataset to %s\n', h5sortFile)
  end
end

main()
