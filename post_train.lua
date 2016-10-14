
require '.'
require 'shortcut'
require 'SelectNetPos'
require 'train'

local Trainer = torch.class('SelectNetPostTrainer', 'SelectNetTrainer')

local function getOpts()
  local cmd = torch.CmdLine()
  cmd:option('--load', '', 'model path')
  cmd:option('--save', 'model.t7', 'save model path')
  cmd:option('--lr', 0.001, 'learning rate')
  cmd:option('--maxEpoch', 30, 'maximum number of epochs')
  cmd:option('--optimMethod', 'SGD', 'optimization algorithm')
  cmd:option('--decay', 1, 'decay learning rate')
  
  local opts = cmd:parse(arg)
  
  return opts
end

function Trainer:main()
  local opts_ = getOpts()
  self.opts = torch.load( opts_.load:sub(1, -3) .. 'state.t7' )
  assert(self.opts.save ~= opts_.save)
  self.opts.load = opts_.load
  self.opts.save = opts_.save
  self.opts.lr = opts_.lr 
  self.opts.maxEpoch = opts_.maxEpoch
  self.opts.optimMethod = opts_.optimMethod
  local opts = self.opts
  
  torch.manualSeed(opts.seed + 1)
  if opts.useGPU then
    require 'cutorch'
    require 'cunn'
    cutorch.manualSeed(opts.seed + 1)
  end
  
  self.trainSize, self.validSize, self.testSize = unpack( DepPosDataIter.getDataSize({opts.train, opts.valid, opts.test}) )
  xprintln('train size = %d, valid size = %d, test size = %d', self.trainSize, self.validSize, self.testSize)
  
  -- local vocabPath = opts.train .. '.tmp.vocab.t7'
  local vocabPath = opts.train .. '.tmp.pos.vocab.t7'
  local recreateVocab = true
  if paths.filep(vocabPath) then
    opts.vocab = torch.load(vocabPath)
    if opts.vocab.ignoreCase == opts.ignoreCase and opts.vocab.freqCut == opts.freqCut and opts.vocab.maxNVocab == opts.maxNVocab then
      recreateVocab = false
      DepPosDataIter.showVocab(opts.vocab)
      print '****load from existing vocab!!!****\n\n'
    end
  end
  if recreateVocab then
    opts.vocab = DepPosDataIter.createVocab(opts.train, opts.ignoreCase, opts.freqCut, opts.maxNVocab)
    torch.save(vocabPath, opts.vocab)
    xprintln('****create vocab from scratch****\n\n')
  end
  
  self.net = SelectNetPos(opts)
  self:showOpts()
  
  xprintln('load from %s ...', opts.load)
  self.net:load(opts.load)
  xprintln('load from %s done!', opts.load)
  
  self.train_all_sents = DepPosDataIter.loadAllSents(opts.vocab, opts.train, opts.maxTrainLen)
  local bestUAS = 0
  local bestModel = torch.FloatTensor(self.net.params:size())
  local timer = torch.Timer()
  
  self.opts.sgdParam = {learningRate = opts.lr}
  local v = self:validConllx(opts.valid)
  print(v)
  bestUAS = v.UAS
  self.net:getModel(bestModel)
  
  for epoch = 1, self.opts.maxEpoch do
    self.iepoch = epoch
    local startTime = timer:time().real
    
    local train_nll, train_perp = self:train()
    xprintln('\nepoch %d TRAIN %f (%f) ', epoch, train_nll, train_perp)
    -- local vret = self:valid(opts.valid)
    local vret = self:validConllx(opts.valid)
    print 'Valid Performance'
    print(vret)
    local endTime = timer:time().real
    xprintln('time spend %s', readableTime(endTime - startTime))
    
    if bestUAS < vret.UAS then
      bestUAS = vret.UAS
      self.net:getModel(bestModel)
      if opts.test and opts.test ~= '' then
        local vret = self:validConllx(opts.test)
        print 'Test Performance'
        print(vret)
      end
    else
      xprintln('UAS on valid not increase! early stopping!')
      break
    end
    
    self.opts.sgdParam.learningRate = self.opts.sgdParam.learningRate * opts_.decay
  end
  
  -- save final model
  self.net:setModel(bestModel)
  opts.sgdParam = nil
  self.net:save(opts.save, true)
  xprintln('model saved at %s', opts.save)
  
  -- show final perform
  local vret = self:validConllx(opts.valid)
  print 'Final Valid Performance'
  print(vret)
  if opts.test and opts.test ~= '' then
    vret = self:validConllx(opts.test)
    print 'Final Test Performance'
    print(vret)
  end
  
end

local function main()
  local trainer = SelectNetPostTrainer()
  trainer:main()
end

if not package.loaded['post_train'] then
  main()
else
  print '[post_train] loaded as package!'
end

