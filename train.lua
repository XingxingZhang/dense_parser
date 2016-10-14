
require '.'
require 'shortcut'
-- require 'SelectNet'
require 'SelectNetPos'
-- require 'DepDataIter'
require 'DepPosDataIter'

local Trainer = torch.class('SelectNetTrainer')

function Trainer:showOpts()
  local tmp_vocab = self.opts.vocab
  self.opts.vocab = nil
  print(self.opts)
  self.opts.vocab = tmp_vocab
end

function Trainer:train()
  local dataIter = DepPosDataIter.createBatchShuffleSort(self.train_all_sents, self.opts.vocab, self.opts.batchSize, 20, true)
  
  local dataSize = self.trainSize
  local curDataSize = 0
  local percent, inc = 0.001, 0.001
  local timer = torch.Timer()
  local sgdParam = self.opts.sgdParam
  local cnt = 0
  local totalLoss = 0
  local totalCnt = 0
  
  for x, x_mask, x_pos, y in dataIter do
    local loss = self.net:trainBatch(x, x_mask, x_pos, y, sgdParam)
    local y_mask = x_mask[{ {2, -1}, {} }]
    
    totalLoss = totalLoss + loss * x:size(2)
    totalCnt = totalCnt + y_mask:sum()
    
    curDataSize = curDataSize + x:size(2)
    local ratio = curDataSize/dataSize
    if ratio >= percent then
      local wps = totalCnt / timer:time().real
      xprint( '\repoch %d %.3f %.4f (%s) / %.2f wps ... ', self.iepoch, ratio, totalLoss/totalCnt, readableTime(timer:time().real), wps )
      percent = math.floor(ratio / inc) * inc
      percent = percent + inc
    end
    
    cnt = cnt + 1
    if cnt % 5 == 0 then
      collectgarbage()
    end
  end
  
  local nll = totalLoss / totalCnt
  return nll, math.exp(nll)
end

function Trainer:valid(validFile)
  local dataIter = DepPosDataIter.createBatchSort(self.opts.vocab, validFile, self.opts.batchSize, 150)
  local totalCnt = 0
  local totalLoss = 0
  local cnt = 0
  local UAS_c, UAS_t = 0, 0
  for x, x_mask, x_pos, y in dataIter do
    local loss, y_preds = self.net:validBatch(x, x_mask, x_pos, y)
    
    totalLoss = totalLoss + loss * x:size(2)
    local y_mask = x_mask[{ {2, -1}, {} }]
    
    local y_p = torch.LongTensor(y:size(1), y:size(2))
    for t = 2, x:size(1) do
      local _, mi = y_preds[t]:max(2)
      if self.opts.useGPU then mi = mi:double() end
      y_p[{ t-1, {} }] = mi
    end
    UAS_c = UAS_c + y:eq(y_p):double():cmul(y_mask):sum()
    UAS_t = UAS_t + y_mask:sum()
    
    totalCnt = totalCnt + y_mask:sum()
    cnt = cnt + 1
    if cnt % 5 == 0 then
      collectgarbage()
    end
  end
  
  local entropy = totalLoss / totalCnt
  local ppl =  torch.exp(entropy)
  
  return {entropy = entropy, ppl = ppl, UAS = (UAS_c / UAS_t)}
end

function Trainer:validConllx(validFile)
  local dataIter = DepPosDataIter.createBatch(self.opts.vocab, validFile, self.opts.batchSize, 150)
  local totalCnt = 0
  local totalLoss = 0
  local cnt = 0
  
  local sents_dep = {}
  local y_tmp = torch.LongTensor(150, self.opts.batchSize)
  for x, x_mask, x_pos, y in dataIter do
    local loss, y_preds = self.net:validBatch(x, x_mask, x_pos, y)
    
    totalLoss = totalLoss + loss * x:size(2)
    local y_mask = x_mask[{ {2, -1}, {} }]
    
    local y_p = y_tmp:resize(y:size(1), y:size(2))
    for t = 2, x:size(1) do
      local _, mi = y_preds[t]:max(2)
      if self.opts.useGPU then mi = mi:double() end
      y_p[{ t-1, {} }] = mi
    end
    
    for i = 1, y_mask:size(2) do
      local slen = y_mask[{ {}, i }]:sum()
      local sent_dep = {}
      for j = 1, slen do
        sent_dep[#sent_dep + 1] = y_p[{ j, i }] - 1
      end
      sents_dep[#sents_dep + 1] = sent_dep
    end
    
    totalCnt = totalCnt + y_mask:sum()
    cnt = cnt + 1
    if cnt % 5 == 0 then
      collectgarbage()
    end
  end
  
  local dep_iter = DepPosDataIter.conllx_iter(validFile)
  local sent_idx = 0
  local sys_out = '__tmp__.dep'
  local fout = io.open(sys_out, 'w')
  for dsent in dep_iter do
    sent_idx = sent_idx + 1
    local sent_dep = sents_dep[sent_idx]
    assert(#sent_dep == #dsent)
    for i, ditem in ipairs(dsent) do
      -- 1	Influential	_	JJ	JJ	_	2	amod	_	_
      fout:write(string.format('%d\t%s\t_\t%s\t_\t_\t%d\tN_A\t_\t_\n', ditem.p1, ditem.wd, ditem.pos, sent_dep[i]))
    end
    fout:write('\n')
  end
  fout:close()
  
  local conllx_eval = self.opts.evalType == 'stanford' and require 'conllx_eval' or require 'conllx2006_eval'
  local LAS, UAS, noPunctLAS, noPunctUAS = conllx_eval.eval(sys_out, validFile)
  
  local entropy = totalLoss / totalCnt
  local ppl =  torch.exp(entropy)
  
  return {entropy = entropy, ppl = ppl, UAS = noPunctUAS}
end

function Trainer:main()
  local model_opts = require 'model_opts'
  local opts = model_opts.getOpts()
  self.opts = opts
  
  self.trainSize, self.validSize, self.testSize = unpack( DepPosDataIter.getDataSize({opts.train, opts.valid, opts.test}) )
  xprintln('train size = %d, valid size = %d, test size = %d', self.trainSize, self.validSize, self.testSize)
  
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
  
  self.train_all_sents = DepPosDataIter.loadAllSents(opts.vocab, opts.train, opts.maxTrainLen)
  local bestUAS = 0
  local bestModel = torch.FloatTensor(self.net.params:size())
  local timer = torch.Timer()
  
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
  local trainer = SelectNetTrainer()
  trainer:main()
end

if not package.loaded['train'] then
  main()
else
  print '[train] loaded as package!'
end


