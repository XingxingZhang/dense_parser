
require '.'
require 'shortcut'
require 'SelectNetPos'
require 'DepPosDataIter'
require 'hdf5'
require 'MLP'

local LabeledTrainer = torch.class('LabeledModelTrainer')

local function getOpts()
  local cmd = torch.CmdLine()
  cmd:option('--mode', 'train', 'two modes: [generate] generate training data; [train] train labeled model')
  -- /disk/scratch/XingxingZhang/dep_parse/experiments/we_select_ft_pos/ft/model_0.0001.std.ft0.t7
  cmd:option('--modelPath', '/disk/scratch/XingxingZhang/dep_parse/experiments/we_select_ft_pos/ft/model_0.001.std.ft0.t7', 'model path')
  cmd:option('--outTrainDataPath', '/disk/scratch/XingxingZhang/dep_parse/experiments/we_select_ft_pos_lbl_pos/label_0.001.std.ft0.pos.h5', 'where will you save the training data')
  cmd:option('--inTrain', '/afs/inf.ed.ac.uk/group/project/img2txt/dep_parser/dataset/train.autopos', 'input training data path')
  cmd:option('--inValid', '/afs/inf.ed.ac.uk/group/project/img2txt/dep_parser/dataset/valid.autopos', 'input validation data path')
  cmd:option('--inTest', '/afs/inf.ed.ac.uk/group/project/img2txt/dep_parser/dataset/test.autopos', 'input test data path')
  cmd:option('--outValid', '', 'valid conllx file from last step')
  cmd:option('--outTest', '', 'test conllx file from last step')
  cmd:option('--language', 'English', 'English or Chinese or Other')
  
  cmd:text('')
  cmd:text('==Options for MLP==')
  cmd:option('--seed', 123, 'random seed')
  cmd:option('--useGPU', false, 'use gpu')
  cmd:option('--snhids', '1460,400,400,45', 'string hidden sizes for each layer')
  cmd:option('--ftype', '|x|', 'type: x, xe, xpe. For example: |x|xe|xpe|')
  cmd:option('--activ', 'relu', 'options: tanh, relu')
  cmd:option('--dropout', 0, 'dropout rate (dropping)')
  cmd:option('--inDropout', 0, 'dropout rate (dropping)')
  cmd:option('--batchNorm', false, 'add batch normalization')
  cmd:option('--maxEpoch', 10, 'max number of epochs')
  cmd:option('--dataset', 
    '/disk/scratch/XingxingZhang/dep_parse/experiments/we_select_ft_pos_lbl_pos/label_0.001.std.ft0.h5', 
    'dataset')
  cmd:option('--batchSize', 256, '')
  cmd:option('--lr', 0.01, '')
  cmd:option('--optimMethod', 'AdaGrad', 'options: SGD, AdaGrad, Adam')
  cmd:option('--save', '/disk/scratch/XingxingZhang/dep_parse/experiments/we_select_ft_pos_lbl_pos/lclassifier_0.001.std.ft0.t7', 'save path')
  
  local opts = cmd:parse(arg)
  
  return opts
end

function LabeledTrainer:showOpts()
  local tmp_vocab = self.opts.vocab
  self.opts.vocab = nil
  print(self.opts)
  self.opts.vocab = tmp_vocab
end

function LabeledTrainer:validConllx(validFile, outputConllFile, defaultLabel)
  xprintln('default label is %s', defaultLabel)
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
  local sys_out = outputConllFile
  local fout = io.open(sys_out, 'w')
  for dsent in dep_iter do
    sent_idx = sent_idx + 1
    local sent_dep = sents_dep[sent_idx]
    assert(#sent_dep == #dsent)
    for i, ditem in ipairs(dsent) do
      -- 1	Influential	_	JJ	JJ	_	2	amod	_	_
      fout:write(string.format('%d\t%s\t_\t_\t%s\t_\t%d\t%s\t_\t_\n', ditem.p1, ditem.wd, ditem.pos, sent_dep[i], defaultLabel))
    end
    fout:write('\n')
  end
  fout:close()
  
  -- local conllx_eval = require 'conllx_eval'
  if  self.opts.evalType == nil then
    self.opts.evalType = 'stanford'
  end
  xprintln('eval type = %s', self.opts.evalType)
  local conllx_eval = self.opts.evalType == 'stanford' and require 'conllx_eval' or require 'conllx2006_eval'
  local LAS, UAS, noPunctLAS, noPunctUAS = conllx_eval.eval(sys_out, validFile)
  
  local entropy = totalLoss / totalCnt
  local ppl =  torch.exp(entropy)
  
  return {entropy = entropy, ppl = ppl, UAS = noPunctUAS}
end

function LabeledTrainer:load(model_path)
  local opts = torch.load( model_path:sub(1, -3) .. 'state.t7' )
  self.opts = opts
  torch.manualSeed(opts.seed)
  if opts.useGPU then
    require 'cutorch'
    require 'cunn'
    cutorch.manualSeed(opts.seed)
  end
  
  assert(opts.vocab ~= nil, 'We must have an existing vocabulary!')
  self.net = SelectNetPos(opts)
  self:showOpts()
  
  xprintln('load from %s ...', model_path)
  self.net:load(model_path)
  xprintln('load from %s done!', model_path)
end

function LabeledTrainer:createTrainData(indDtaPaths, outDataPath, language)
  self.rel_vocab = DepPosDataIter.createDepRelVocab(indDtaPaths.train)
  print(self.rel_vocab)
  local h5out = hdf5.open(outDataPath, 'w')
  
  local function generateSplit(slabel, infile, batchSize, maxlen)
    local gxdata = string.format('/%s/x', slabel)
    local gydata = string.format('/%s/y', slabel)
    local gxedata = string.format('/%s/xe', slabel)
    local gxpedata = string.format('/%s/xpe', slabel)
    
    local xOpt = hdf5.DataSetOptions()
    xOpt:setChunked(1024*10, self.opts.nhid * 4)
    xOpt:setDeflate()
    
    local xeOpt = hdf5.DataSetOptions()
    xeOpt:setChunked(1024*10, self.opts.nin * 2)
    xeOpt:setDeflate()
    
    local xpeOpt = hdf5.DataSetOptions()
    xpeOpt:setChunked(1024*10, self.opts.npin * 2)
    xpeOpt:setDeflate()
    
    local yOpt = hdf5.DataSetOptions()
    yOpt:setChunked(1024*10)
    yOpt:setDeflate()
    
    local isFirst = true
    local diter = DepPosDataIter.createBatchLabel(self.opts.vocab, self.rel_vocab, infile, batchSize, maxlen)
    local cnt = 0
    for x, x_mask, x_pos, y, sent_rels, sent_ori_rels in diter do
      self.net:validBatch(x, x_mask, x_pos, y)
      local dsize = x_mask:sum() - x_mask:size(2)
      assert(dsize == y:ne(0):sum(), 'size should be the same')
      local x_input = torch.zeros(dsize, self.opts.nhid * 4):float()
      local y_output = torch.zeros(dsize):int()
      local x_input_emb = torch.zeros(dsize, self.opts.nin * 2):float()
      local x_input_pos_emb = torch.zeros(dsize, self.opts.npin * 2):float()
      
      -- self.mod_map.forward_lookup
      -- self.mod_map.forward_pos_lookup
      local x_emb = self.net.mod_map.forward_lookup:forward(x)
      local x_pos_emb = self.net.mod_map.forward_pos_lookup:forward(x_pos)
      
      -- bs x seqlen x nhid
      -- self.net.all_fwd_bak_hs
      local example_cnt = 0
      for i, sent_rel in ipairs(sent_rels) do
        assert(x_mask[{ {}, i }]:sum() == #sent_rel + 1, 'MUST be the same length')
        for j, rel_id in ipairs(sent_rel) do
          local cur_id = j + 1
          local parent_id = y[{ j, i }]
          local cur_a = self.net.all_fwd_bak_hs[{ i, cur_id, {} }]
          local parent_a = self.net.all_fwd_bak_hs[{ i, parent_id, {} }]
          example_cnt = example_cnt + 1
          x_input[{ example_cnt, {1, 2 * self.opts.nhid} }] = cur_a:float()
          x_input[{ example_cnt, {2 * self.opts.nhid + 1, 4 * self.opts.nhid} }] = parent_a:float()
          y_output[{ example_cnt }] = rel_id
          
          local cur_emb = x_emb[{ cur_id, i, {} }]
          local parent_emb = x_emb[{ parent_id, i, {} }]
          local cur_pos_emb = x_pos_emb[{ cur_id, i, {} }]
          local parent_pos_emb = x_pos_emb[{ parent_id, i, {} }]
          x_input_emb[{ example_cnt, {1, self.opts.nin} }] = cur_emb:float()
          x_input_emb[{ example_cnt, {self.opts.nin + 1, 2*self.opts.nin} }] = parent_emb:float()
          x_input_pos_emb[{ example_cnt, {1, self.opts.npin} }] = cur_pos_emb:float()
          x_input_pos_emb[{ example_cnt, {self.opts.npin + 1, 2*self.opts.npin} }] = parent_pos_emb:float()
        end
      end
      
      if isFirst then
        h5out:write(gxdata, x_input, xOpt)
        h5out:write(gydata, y_output, yOpt)
        
        h5out:write(gxedata, x_input_emb, xeOpt)
        h5out:write(gxpedata, x_input_pos_emb, xpeOpt)
        
        isFirst = false
      else
        h5out:append(gxdata, x_input, xOpt)
        h5out:append(gydata, y_output, yOpt)
        
        h5out:append(gxedata, x_input_emb, xeOpt)
        h5out:append(gxpedata, x_input_pos_emb, xpeOpt)
      end
      
      cnt = cnt + 1
      if cnt % 5 == 0 then
        collectgarbage()
      end
      
      if cnt % 10 == 0 then
        xprint('cnt = %d\n', cnt)
      end
    end
    
    print( 'toally ' .. cnt )
  end
  
  local predictValidFile = outDataPath .. '.valid.conllx'
  local predictTestFile = outDataPath .. '.test.conllx'
  print(indDtaPaths)
  
  local dlabel = self.rel_vocab.idx2rel[1]
  xprintln('the default dependency label is %s\n', dlabel)
  
  if indDtaPaths.outvalid == '' then
    assert(language == 'English' or language == 'Chinese' or language == 'Other')
    --[[
    local dlabel
    if language == 'English' then
      dlabel = 'pobj'
    elseif language == 'Chinese' then
      dlabel = 'ROOT'
    end
    --]]
    
    self:validConllx(indDtaPaths.valid, predictValidFile, dlabel)
    self:validConllx(indDtaPaths.test, predictTestFile, dlabel)
  else
    -- predictValidFile = indDtaPaths.outvalid
    -- predictTestFile = indDtaPaths.outtest
    assert(language == 'English' or language == 'Chinese' or language == 'Other')
    if language == 'English' then
      os.execute( string.format('cp %s %s', indDtaPaths.outvalid, predictValidFile) )
      os.execute( string.format('cp %s %s', indDtaPaths.outtest, predictTestFile) )
    else
      local replaceField = require 'replace_conllx_field'
      replaceField.replace(indDtaPaths.outvalid, predictValidFile, 8, dlabel)
      replaceField.replace(indDtaPaths.outtest, predictTestFile, 8, dlabel)
      xprintln('change field 8 to %s', dlabel)
    end
    
    if  self.opts.evalType == nil then
      self.opts.evalType = 'stanford'
    end
    xprintln('eval type = %s', self.opts.evalType)
    local conllx_eval = self.opts.evalType == 'stanford' and require 'conllx_eval' or require 'conllx2006_eval'
    -- local conllx_eval = require 'conllx_eval'
    print '===Valid==='
    conllx_eval.eval(predictValidFile, indDtaPaths.valid)
    print '===Test==='
    conllx_eval.eval(predictTestFile, indDtaPaths.test)
  end
  
  assert(language == 'English' or language == 'Chinese' or language == 'Other')
  local maxTrainLen = language == 'English' and 100 or 140
  if language == 'Other' then
    maxTrainLen = 110
  end
  if self.opts.maxTrainLen ~= nil then
    maxTrainLen = self.opts.maxTrainLen
    print('maxTrainLen = ', maxTrainLen)
  end
  
  generateSplit('predict_valid', predictValidFile, self.opts.batchSize, 999999)
  generateSplit('predict_test', predictTestFile, self.opts.batchSize, 999999)
  generateSplit('valid', indDtaPaths.valid, self.opts.batchSize, 999999)
  generateSplit('test', indDtaPaths.test, self.opts.batchSize, 999999)
  generateSplit('train', indDtaPaths.train, self.opts.batchSize, maxTrainLen)
  
  h5out:close()
end

local DataIter = {}
function DataIter.getNExamples(dataPath, label)
  local h5in = hdf5.open(dataPath, 'r')
  local x_data = h5in:read(string.format('/%s/x', label))
  local N = x_data:dataspaceSize()[1]
  
  return N
end

function DataIter.createBatch(dataPath, label, batchSize, ftype)
  local h5in = hdf5.open(dataPath, 'r')
  
  local x_data = h5in:read(string.format('/%s/x', label))
  local xe_data = h5in:read(string.format('/%s/xe', label))
  local xpe_data = h5in:read(string.format('/%s/xpe', label))
  
  local y_data = h5in:read(string.format('/%s/y', label))
  local N = x_data:dataspaceSize()[1]
  local x_width = x_data:dataspaceSize()[2]
  local xe_width = xe_data:dataspaceSize()[2]
  local xpe_width = xpe_data:dataspaceSize()[2]
  
  local istart = 1
  
  return function()
    if istart <= N then
      local iend = math.min(istart + batchSize - 1, N)
      -- local x = x_data:partial({istart, iend}, {1, x_width})
      local y = y_data:partial({istart, iend})
      
      local widths = {x_width}
      local xdatas = {x_data}
      if ftype:find('|xe|') then
        widths[#widths + 1] = xe_width
        xdatas[#xdatas + 1] = xe_data
      end
      if ftype:find('|xpe|') then
        widths[#widths + 1] = xpe_width
        xdatas[#xdatas + 1] = xpe_data
      end
      
      local width = 0
      for _, w in ipairs(widths) do width = width + w end
      local x = torch.zeros(y:size(1), width):float()
      local s = 0
      for i, w in ipairs(widths) do
        x[{ {}, {s + 1, s + w} }] = xdatas[i]:partial({istart, iend}, {1, w})
        s = s + w
      end
      
      istart = iend + 1
      
      return x, y
    else
      h5in:close()
    end
  end
end


local RndBatcher = torch.class('RandomBatcher')
function RndBatcher:__init(h5in, x_data, xe_data, xpe_data, y_data, bufSize, ftype)
  self.h5in = h5in
  self.x_data = x_data
  self.xe_data = xe_data
  self.xpe_data = xpe_data
  self.y_data = y_data
  self.bufSize = bufSize
  self.N = x_data:dataspaceSize()[1]
  self.x_width = x_data:dataspaceSize()[2]
  self.xe_width = xe_data:dataspaceSize()[2]
  self.xpe_width = xpe_data:dataspaceSize()[2]
  
  self.ftype = ftype
  
  self.istart = 1
  self.idx_chunk = 1
  self.chunk_size = 0
end

function RndBatcher:nextChunk()
  if self.istart <= self.N then
    local iend = math.min( self.istart + self.bufSize - 1, self.N )
    self.x_chunk = self.x_data:partial({self.istart, iend}, {1, self.x_width})
    self.xe_chunk = self.xe_data:partial({self.istart, iend}, {1, self.xe_width})
    self.xpe_chunk = self.xpe_data:partial({self.istart, iend}, {1, self.xpe_width})
    
    self.y_chunk = self.y_data:partial({self.istart, iend})
    
    self.chunk_size = iend - self.istart + 1
    
    self.istart = iend + 1
    
    return true
  else
    return false
  end
end

function RndBatcher:nextBatch(batchSize)
  if self.idx_chunk > self.chunk_size then
    if self:nextChunk() then
      self.idx_chunk = 1
      self.idxs_chunk = torch.randperm(self.chunk_size):long()
    else
      return nil
    end
  end
  
  local iend = math.min( self.idx_chunk + batchSize - 1, self.chunk_size )
  local idxs = self.idxs_chunk[{ {self.idx_chunk, iend} }]
  
  local y = self.y_chunk:index(1, idxs)
  
  local xs = {}
  local widths = {}
  local width = 0
  if self.ftype:find('|x|') then
    local x = self.x_chunk:index(1, idxs)
    width = width + self.x_width
    widths[#widths + 1] = self.x_width
    xs[#xs + 1] = x
  end
  
  if self.ftype:find('|xe|') then
    local xe = self.xe_chunk:index(1, idxs)
    width = width + self.xe_width
    widths[#widths + 1] = self.xe_width
    xs[#xs + 1] = xe
  end
  
  if self.ftype:find('|xpe|') then
    local xpe = self.xpe_chunk:index(1, idxs)
    width = width + self.xpe_width
    widths[#widths + 1] = self.xpe_width
    xs[#xs + 1] = xpe
  end
  
  local x_ = torch.zeros(y:size(1), width):float()
  local s = 0
  for i, w in ipairs(widths) do
    x_[{ {}, {s+1, s+w} }] = xs[i]
    s = s + w
  end
  
  self.idx_chunk = iend + 1
  
  return x_, y
end

function DataIter.createBatchShuffle(dataPath, label, batchSize, ftype)
  local h5in = hdf5.open(dataPath, 'r')
  
  local x_data = h5in:read(string.format('/%s/x', label))
  local xe_data = h5in:read(string.format('/%s/xe', label))
  local xpe_data = h5in:read(string.format('/%s/xpe', label))
  
  local y_data = h5in:read(string.format('/%s/y', label))
  
  local bufSize = 1000 * batchSize
  local rnd_batcher = RandomBatcher(h5in, x_data, xe_data, xpe_data, y_data, bufSize, ftype)
  
  return function()
    return rnd_batcher:nextBatch(batchSize)
  end
end

function LabeledTrainer:train_label()
  local dataIter = DataIter.createBatchShuffle(self.classifier_opts.dataset, 'train', 
    self.classifier_opts.batchSize, self.classifier_opts.ftype)
  --[[
  local dataIter = DataIter.createBatch(self.classifier_opts.dataset, 'train', 
    self.classifier_opts.batchSize, self.classifier_opts.ftype)
    --]]
  
  local dataSize = DataIter.getNExamples(self.classifier_opts.dataset, 'train')
  local percent, inc = 0.001, 0.001
  local timer = torch.Timer()
  -- local sgdParam = {learningRate = opts.curLR}
  local sgdParam = self.classifier_opts.sgdParam
  local cnt = 0
  local totalLoss = 0
  local totalCnt = 0
  for x, y in dataIter do
    local loss = self.mlp:trainBatch(x, y, sgdParam)
    totalLoss = totalLoss + loss * x:size(1)
    totalCnt = totalCnt + x:size(1)
    
    local ratio = totalCnt/dataSize
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
  
  return totalLoss / totalCnt
end

function LabeledTrainer:valid_label(label)
  local dataIter = DataIter.createBatch(self.classifier_opts.dataset, label, 
    self.classifier_opts.batchSize, self.classifier_opts.ftype)
  
  local cnt = 0
  local correct, total = 0, 0
  for x, y in dataIter do
    local correct_, total_ = self.mlp:validBatch(x, y)
    correct = correct + correct_
    total = total + total_
    cnt = cnt + 1
    if cnt % 5 == 0 then collectgarbage() end
  end
  
  return correct, total
end

function LabeledTrainer:valid_label_conllx(label, conllx_file, gold_file)
  local dataIter = DataIter.createBatch(self.classifier_opts.dataset, label, 
    self.classifier_opts.batchSize, self.classifier_opts.ftype)
  
  local cnt = 0
  local correct, total = 0, 0
  local lbl_idxs = {}
  for x, y in dataIter do
    local correct_, total_, y_pred = self.mlp:validBatch(x, y)
    correct = correct + correct_
    total = total + total_
    cnt = cnt + 1
    if cnt % 5 == 0 then collectgarbage() end
    
    local y_pred_ = y_pred:view(-1)
    for i = 1, y_pred_:size(1) do
      lbl_idxs[#lbl_idxs + 1] = y_pred_[i]
    end
  end
  
  local ilbl = 0
  local conllx_file_out = conllx_file .. '.out'
  
  -- begin
  local dep_iter = DepPosDataIter.conllx_iter(conllx_file)
  local sys_out = conllx_file_out
  local fout = io.open(sys_out, 'w')
  for dsent in dep_iter do
    for _, ditem in ipairs(dsent) do
      -- 1	Influential	_	JJ	JJ	_	2	amod	_	_
      ilbl = ilbl + 1
      local lbl = self.rel_vocab.idx2rel[ lbl_idxs[ilbl] ]
      fout:write( string.format('%d\t%s\t_\t_\t%s\t_\t%d\t%s\t_\t_\n', ditem.p1, ditem.wd, ditem.pos, ditem.p2, lbl) )
    end
    fout:write('\n')
  end
  fout:close()
  -- end
  
  -- local conllx_eval = require 'conllx_eval'
  local conllx_eval
  xprintln('language = %s', self.classifier_opts.language)
  if self.classifier_opts.language == 'Other' then
    conllx_eval = require 'conllx2006_eval'
  else
    conllx_eval = require 'conllx_eval'
  end
  
  -- xprintln('eval type = %s', self.opts.evalType)
  local LAS, UAS, noPunctLAS, noPunctUAS = conllx_eval.eval(sys_out, gold_file)
  
  return {LAS = noPunctLAS, UAS = noPunctUAS}
end

function LabeledTrainer:trainLabeledClassifier(opts)
  torch.manualSeed(opts.seed)
  if opts.useGPU then
    require 'cutorch'
    require 'cunn'
    cutorch.manualSeed(opts.seed)
  end
  local mlp = MLP(opts)
  opts.sgdParam = {learningRate = opts.lr}
  opts.curLR = opts.lr
  print(opts)
  
  self.classifier_opts = opts
  self.mlp = mlp
  
  local timer = torch.Timer()
  local bestAcc = 0
  local bestModel = torch.FloatTensor(mlp.params:size())
  local bestLAS = 0
  
  self.rel_vocab = DepPosDataIter.createDepRelVocab(opts.inTrain)
  xprintln('load rel_vocab done!')
  self.predictValidFile = opts.dataset .. '.valid.conllx'
  self.predictTestFile = opts.dataset .. '.test.conllx'
  
  for epoch = 1, opts.maxEpoch do
    self.iepoch = epoch
    -- EPOCH_INFO = string.format('epoch %d', epoch)
    local startTime = timer:time().real
    local trainCost = self:train_label()
    xprint('\repoch %d TRAIN nll %f ', epoch, trainCost)
    -- local validCor, validTot = valid(mlp, 'valid', opts)
    local validCor, validTot = self:valid_label('valid')
    local validAcc = validCor/validTot
    xprint('VALID %d/%d = %f ', validCor, validTot, validAcc)
    local endTime = timer:time().real
    xprintln('lr = %.4g (%s)', opts.curLR, readableTime(endTime - startTime))
    
    local v_ret = self:valid_label_conllx('predict_valid', self.predictValidFile, self.classifier_opts.inValid)
    print '==Valid Perf=='
    print(v_ret)
    print '\n'
    
    if v_ret.LAS > bestLAS then
      bestLAS = v_ret.LAS
      mlp:getModel(bestModel)
      
      local t_ret = self:valid_label_conllx('predict_test', self.predictTestFile, self.classifier_opts.inTest)
      print '==Test Perf=='
      print(t_ret)
      print '\n'
    end
  end
  
  mlp:setModel(bestModel)
  opts.sgdParam = nil
  mlp:save(opts.save, true)
  xprintln('model saved at %s', opts.save)
  
  local v_ret = self:valid_label_conllx('predict_valid', self.predictValidFile, self.classifier_opts.inValid)
  print '==Valid Perf=='
  print(v_ret)
  print '\n'
  
  local t_ret = self:valid_label_conllx('predict_test', self.predictTestFile, self.classifier_opts.inTest)
  print '==Test Perf=='
  print(t_ret)
  print '\n'
end

local function main()
  local opts = getOpts()
  local trainer = LabeledModelTrainer()
  if opts.mode == 'generate' then
    xprintln('This is generate mode!')
    trainer:load(opts.modelPath)
    local inDataPaths = {train = opts.inTrain, valid = opts.inValid, test = opts.inTest}
    inDataPaths.outvalid = opts.outValid
    inDataPaths.outtest = opts.outTest
    trainer:createTrainData(inDataPaths, opts.outTrainDataPath, opts.language)
    xprintln('create training data done!')
  elseif opts.mode == 'train' then
    xprintln('This is train mode!')
    trainer:trainLabeledClassifier(opts)
    xprintln('Training done!')
  else
    error('only support [generate] and [train] mode')
  end
end

main()
