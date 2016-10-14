
require 'hdf5'
require 'torch'
include '../utils/shortcut.lua'
require 'deptreeutils'

local function getOpts()
  local cmd = torch.CmdLine()
  cmd:text('== convert testset dependency trees to hdf5 format ==')
  cmd:text()
  cmd:text('Options:')
  cmd:option('--vocab', '', 'vocabulary file (.t7)')
  cmd:option('--test', '', 'test text file (plain text)')
  cmd:option('--testdataset', '', 'the output file of testset (.h5)')
  
  cmd:option('--bidirectional', false, 'create bidirectional model')
  
  return cmd:parse(arg)
end

local function main()
  local opts = getOpts()
  print(opts)
  
  print('load vocab ...')
  local vocab = torch.load(opts.vocab)
  print('load vocab done!')
  local h5out = hdf5.open(opts.testdataset, 'w')
  if opts.bidirectional then
    DepTreeUtils.deptree2hdf5Bidirectional(opts.test, h5out, 'test', vocab, 123456789)
  else
    DepTreeUtils.deptree2hdf5(opts.test, h5out, 'test', vocab, 123456789)
  end
  print('create testset done!')
  h5out:close()
end

main()
