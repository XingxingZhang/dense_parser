
require 'hdf5'
require 'torch'
include '../utils/shortcut.lua'
require 'deptreeutils'

local function getOpts()
  local cmd = torch.CmdLine()
  cmd:text('== convert CoNLL-X dependency trees to hdf5 format ==')
  cmd:text()
  cmd:text('Options:')
  cmd:option('--train', '', 'train CoNLL X file')
  cmd:option('--valid', '', 'valid CoNLL X file')
  cmd:option('--test', '', 'test CoNLL X file')
  cmd:option('--dataset', '', 'the resulting dataset (.h5)')
  cmd:option('--freq', 0, 'words less than or equals to \"freq\" times will be replaced with UNK token')
  cmd:option('--ignoreCase', false, 'case will be ignored')
  cmd:option('--normalizeNumber', false, 'normalize numbers to <num>')
  cmd:option('--keepFreq', false, 'keep frequency information during creating vocabulary')
  cmd:option('--maxLen', 100, 'sentences longer than maxlen will be ignored!')
  cmd:option('--sort', 0, '0: no sorting of the training data; -1: sort training data by their length; k (k > 0): sort the consecutive k batches by their length')
  cmd:option('--batchSize', 64, 'batch size when --sort > 0 or --sort == -1')
  
  cmd:option('--bidirectional', false, 'create bidirectional model')
  
  return cmd:parse(arg)
end

local function main()
  local opts = getOpts()
  print(opts)
  local vocab = DepTreeUtils.createVocabCoNLLX(opts.train, opts.freq, opts.ignoreCase, opts.keepFreq, opts.normalizeNumber)
  assert(opts.dataset:ends('.h5'), 'dataset must be hdf5 file .h5')
  local dataPrefix = opts.dataset:sub(1, -4)
  local vocabPath = dataPrefix .. '.vocab.t7'
  printf('save vocab to %s\n', vocabPath)
  torch.save(vocabPath, vocab)
  
  local h5out = hdf5.open(opts.dataset, 'w')
  if opts.bidirectional then
    DepTreeUtils.conllx2hdf5Bidirectional(opts.train, h5out, 'train', vocab, opts.maxLen)
  else
    DepTreeUtils.conllx2hdf5(opts.train, h5out, 'train', vocab, opts.maxLen)
  end
  print('create training set done!')
  if opts.bidirectional then
    DepTreeUtils.conllx2hdf5Bidirectional(opts.valid, h5out, 'valid', vocab, opts.maxLen)
  else
    DepTreeUtils.conllx2hdf5(opts.valid, h5out, 'valid', vocab, opts.maxLen)
  end
  print('create validating set done!')
  if opts.bidirectional then
    DepTreeUtils.conllx2hdf5Bidirectional(opts.test, h5out, 'test', vocab, opts.maxLen)
  else
    DepTreeUtils.conllx2hdf5(opts.test, h5out, 'test', vocab, opts.maxLen)
  end
  print('create testing set done!')
  h5out:close()
  printf('save dataset to %s\n', opts.dataset)
  
  if opts.sort ~= 0 then
    assert(opts.sort == -1 or opts.sort > 0, 'valid values [0, -1, > 0]')
    print '========begin to sort dataset========'
    require 'sorthdf5'
    local h5sorter = nil
    local mid = opts.sort == -1 and 'sort' or string.format('sort%d', opts.sort)
    local h5sortFile = opts.dataset:sub(1, -4) .. string.format('.%s.h5', mid)
    local h5sortVocabFile = opts.dataset:sub(1, -4) .. string.format('.%s.vocab.t7', mid)
    local cmd = string.format('cp %s %s', vocabPath, h5sortVocabFile)
    print(cmd)
    os.execute(cmd)
    if opts.bidirectional then
      require 'sorthdf5bid'
      h5sorter = SortHDF5Bidirectional(opts.dataset, h5sortFile, h5sortVocabFile)
    else
      h5sorter = SortHDF5(opts.dataset, h5sortFile, h5sortVocabFile)
    end
    h5sorter:sortHDF5(opts.sort, opts.batchSize)
    printf('save dataset to %s\n', h5sortFile)
  end
  
end

main()
