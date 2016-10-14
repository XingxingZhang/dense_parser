
require '.'
require 'shortcut'
require 'SelectNetPos'
require 'DepPosDataIter'
require 'PostDepGraph'
require 'ChuLiuEdmonds'
require 'Eisner'

local MST = torch.class('MSTPostProcessor')

function MST:showOpts()
  local tmp_vocab = self.opts.vocab
  self.opts.vocab = nil
  print(self.opts)
  self.opts.vocab = tmp_vocab
end

function MST:load(modelPath)
  self.opts = torch.load( modelPath:sub(1, -3) .. 'state.t7' )
  local opts = self.opts
  
  torch.manualSeed(opts.seed + 1)
  if opts.useGPU then
    require 'cutorch'
    require 'cunn'
    cutorch.manualSeed(opts.seed + 1)
  end
  
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
  assert(not recreateVocab, 'you should load existing vocabulary')
  
  self.net = SelectNetPos(opts)
  self:showOpts()
  
  xprintln('load from %s ...', modelPath)
  self.net:load(modelPath)
  xprintln('load from %s done!', modelPath)
end

function MST:validConllx(validFile, outConllxFile)
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
  
  outConllxFile = outConllxFile or '__tmp__.dep'
  
  local dep_iter = DepPosDataIter.conllx_iter(validFile)
  local sent_idx = 0
  local sys_out = outConllxFile
  local fout = io.open(sys_out, 'w')
  for dsent in dep_iter do
    sent_idx = sent_idx + 1
    local sent_dep = sents_dep[sent_idx]
    assert(#sent_dep == #dsent)
    for i, ditem in ipairs(dsent) do
      -- '%d\t%s\t_\t_\t%s\t_\t%d\t%s\t_\t_\n'
      -- 1	Influential	_	JJ	JJ	_	2	amod	_	_
      fout:write(string.format('%d\t%s\t_\t_\t%s\t_\t%d\tprep\t_\t_\n', ditem.p1, ditem.wd, ditem.pos, sent_dep[i]))
    end
    fout:write('\n')
  end
  fout:close()
  
  -- local conllx_eval = require 'conllx_eval'
  local conllx_eval = self.opts.evalType == 'stanford' and require 'conllx_eval' or require 'conllx2006_eval'
  local LAS, UAS, noPunctLAS, noPunctUAS = conllx_eval.eval(sys_out, validFile)
  
  local entropy = totalLoss / totalCnt
  local ppl =  torch.exp(entropy)
  
  return {entropy = entropy, ppl = ppl, UAS = noPunctUAS}
end

function MST:validWithMSTPost(validFile, outConllxFile)
  local dataIter = DepPosDataIter.createBatch(self.opts.vocab, validFile, self.opts.batchSize, 150)
  local totalCnt = 0
  local totalLoss = 0
  local cnt = 0
  
  local sents_dep = {}
  local sents_graph = {}
  local y_tmp = torch.LongTensor(150, self.opts.batchSize)
  for x, x_mask, x_pos, y in dataIter do
    local loss, y_preds = self.net:validBatch(x, x_mask, x_pos, y)
    
    totalLoss = totalLoss + loss * x:size(2)
    local y_mask = x_mask[{ {2, -1}, {} }]
    
    local y_p = y_tmp:resize(y:size(1), y:size(2))
    -- WARNING: y_preds start from 2!
    for t = 2, x:size(1) do
      local _, mi = y_preds[t]:max(2)
      if self.opts.useGPU then mi = mi:double() end
      y_p[{ t-1, {} }] = mi
    end
    
    for i = 1, y_mask:size(2) do
      local slen = y_mask[{ {}, i }]:sum()
      local sent_dep = {}
      local sent_graph = {}
      for j = 1, slen do
        sent_dep[#sent_dep + 1] = y_p[{ j, i }] - 1
        local tmp = y_preds[j+1][{ i, {1, slen + 1} }]:double()
        sent_graph[j] = tmp
      end
      sents_dep[#sents_dep + 1] = sent_dep
      sents_graph[#sents_graph + 1] = sent_graph
    end
    
    totalCnt = totalCnt + y_mask:sum()
    cnt = cnt + 1
    if cnt % 5 == 0 then
      collectgarbage()
    end
  end
  
  outConllxFile = outConllxFile or '__tmp__.dep'
  
  local dep_iter = DepPosDataIter.conllx_iter(validFile)
  local sent_idx = 0
  local connected_count = 0
  local sys_out = outConllxFile
  local fout = io.open(sys_out, 'w')
  for dsent in dep_iter do
    sent_idx = sent_idx + 1
    local sent_dep = sents_dep[sent_idx]
    assert(#sent_dep == #dsent)
    local sent_graph = sents_graph[sent_idx]
    assert(#sent_graph == #dsent)
    
    local new_dsent = {}
    for i, ditem in ipairs(dsent) do
      local new_ditem = {p1 = ditem.p1, wd = ditem.wd, pos = ditem.pos, p2 = sent_dep[i]}
      new_dsent[#new_dsent + 1] = new_ditem
    end
    
    -- check connectivity
    local dgraph = PostDepGraph(new_dsent)
    if not dgraph:checkConnectivity() then
      local N = #sent_graph + 1
      local edges = {}
      for i, sp in ipairs(sent_graph) do
        for j = 1, sp:size(1) do
          edges[#edges + 1] = {j, i+1, sp[j]}
        end
      end
      -- run ChuLiuEdmonds
      local cle = ChuLiuEdmonds()
      cle:load(N, edges)
      local _, selectedEdges = cle:solve(1, N)
      table.sort(selectedEdges, function(a, b) return a.v < b.v end)
      for i, ditem in ipairs(new_dsent) do
        local edge = selectedEdges[i]
        assert(edge.v == i+1)
        ditem.p2 = edge.u - 1
        ditem.p1 = edge.v - 1
      end
      
      local dgraph2 = PostDepGraph(new_dsent)
      assert(dgraph2:checkConnectivity())
    else
      connected_count = connected_count + 1
    end
    
    for i, ditem in ipairs(new_dsent) do
      -- '%d\t%s\t_\t_\t%s\t_\t%d\t%s\t_\t_\n'
      -- 1	Influential	_	JJ	JJ	_	2	amod	_	_
      fout:write(string.format('%d\t%s\t_\t_\t%s\t_\t%d\tprep\t_\t_\n', ditem.p1, ditem.wd, ditem.pos, ditem.p2))
    end
    fout:write('\n')
  end
  fout:close()
  printf('%d/%d = %f are connected graph\n', connected_count, sent_idx, connected_count/sent_idx)
  
  -- local conllx_eval = require 'conllx_eval'
  printf('evalType = %s\n', self.opts.evalType)
  local conllx_eval = self.opts.evalType == 'stanford' and require 'conllx_eval' or require 'conllx2006_eval'
  local LAS, UAS, noPunctLAS, noPunctUAS = conllx_eval.eval(sys_out, validFile)
  
  local entropy = totalLoss / totalCnt
  local ppl =  torch.exp(entropy)
  
  return {entropy = entropy, ppl = ppl, UAS = noPunctUAS}
end


function MST:validWithMSTPostEisner(validFile, outConllxFile)
  local dataIter = DepPosDataIter.createBatch(self.opts.vocab, validFile, self.opts.batchSize, 150)
  local totalCnt = 0
  local totalLoss = 0
  local cnt = 0
  
  local sents_dep = {}
  local sents_graph = {}
  local y_tmp = torch.LongTensor(150, self.opts.batchSize)
  for x, x_mask, x_pos, y in dataIter do
    local loss, y_preds = self.net:validBatch(x, x_mask, x_pos, y)
    
    totalLoss = totalLoss + loss * x:size(2)
    local y_mask = x_mask[{ {2, -1}, {} }]
    
    local y_p = y_tmp:resize(y:size(1), y:size(2))
    -- WARNING: y_preds start from 2!
    for t = 2, x:size(1) do
      local _, mi = y_preds[t]:max(2)
      if self.opts.useGPU then mi = mi:double() end
      y_p[{ t-1, {} }] = mi
    end
    
    for i = 1, y_mask:size(2) do
      local slen = y_mask[{ {}, i }]:sum()
      local sent_dep = {}
      local sent_graph = {}
      for j = 1, slen do
        sent_dep[#sent_dep + 1] = y_p[{ j, i }] - 1
        local tmp = y_preds[j+1][{ i, {1, slen + 1} }]:double()
        sent_graph[j] = tmp
      end
      sents_dep[#sents_dep + 1] = sent_dep
      sents_graph[#sents_graph + 1] = sent_graph
    end
    
    totalCnt = totalCnt + y_mask:sum()
    cnt = cnt + 1
    if cnt % 5 == 0 then
      collectgarbage()
    end
  end
  
  outConllxFile = outConllxFile or '__tmp__.dep'
  
  local dep_iter = DepPosDataIter.conllx_iter(validFile)
  local sent_idx = 0
  local connected_count = 0
  local sys_out = outConllxFile
  local fout = io.open(sys_out, 'w')
  for dsent in dep_iter do
    sent_idx = sent_idx + 1
    local sent_dep = sents_dep[sent_idx]
    assert(#sent_dep == #dsent)
    local sent_graph = sents_graph[sent_idx]
    assert(#sent_graph == #dsent)
    
    local new_dsent = {}
    for i, ditem in ipairs(dsent) do
      local new_ditem = {p1 = ditem.p1, wd = ditem.wd, pos = ditem.pos, p2 = sent_dep[i]}
      new_dsent[#new_dsent + 1] = new_ditem
    end
    
    -- check connectivity
    local dgraph = PostDepGraph(new_dsent)
    if not (dgraph:checkConnectivity() and dgraph:isProjective()) then
      local N = #sent_graph + 1
      local edges = {}
      for i, sp in ipairs(sent_graph) do
        for j = 1, sp:size(1) do
          edges[#edges + 1] = {j, i+1, sp[j]}
        end
      end
      -- run Eisner's algorithm
      local eisner = Eisner()
      eisner:load(N, edges)
      local _, selectedEdges = eisner:solve()
      table.sort(selectedEdges, function(a, b) return a.v < b.v end)
      for i, ditem in ipairs(new_dsent) do
        local edge = selectedEdges[i]
        assert(edge.v == i+1)
        ditem.p2 = edge.u - 1
        ditem.p1 = edge.v - 1
      end
      
      -- local dgraph2 = PostDepGraph(new_dsent)
      -- assert(dgraph2:checkConnectivity())
    else
      connected_count = connected_count + 1
    end
    
    for i, ditem in ipairs(new_dsent) do
      -- '%d\t%s\t_\t_\t%s\t_\t%d\t%s\t_\t_\n'
      -- 1	Influential	_	JJ	JJ	_	2	amod	_	_
      fout:write(string.format('%d\t%s\t_\t_\t%s\t_\t%d\tprep\t_\t_\n', ditem.p1, ditem.wd, ditem.pos, ditem.p2))
    end
    fout:write('\n')
  end
  fout:close()
  printf('%d/%d = %f are projective trees\n', connected_count, sent_idx, connected_count/sent_idx)
  
  -- local conllx_eval = require 'conllx_eval'
  printf('evalType = %s\n', self.opts.evalType)
  local conllx_eval = self.opts.evalType == 'stanford' and require 'conllx_eval' or require 'conllx2006_eval'
  local LAS, UAS, noPunctLAS, noPunctUAS = conllx_eval.eval(sys_out, validFile)
  
  local entropy = totalLoss / totalCnt
  local ppl =  torch.exp(entropy)
  
  return {entropy = entropy, ppl = ppl, UAS = noPunctUAS}
end

local function getOpts()
  local cmd = torch.CmdLine()
  cmd:option('--modelPath', '/disk/scratch/XingxingZhang/dep_parse/experiments/we_select_ft_pos/ft/model_0.001.std.ft0.t7', 'model path')
  cmd:option('--validout', 'valid', 'output conllx file for validation set')
  cmd:option('--testout', 'test', 'output conllx file for test set')
  cmd:option('--mstalg', 'ChuLiuEdmonds', 'MST algorithm: ChuLiuEdmonds or Eisner')
  
  return cmd:parse(arg)
end

local function main()
  local opts = getOpts()
  -- local modelPath = '/disk/scratch/XingxingZhang/dep_parse/experiments/we_select_ft_pos/ft/model_0.001.std.ft0.t7'
  -- local modelPath = '/disk/scratch/XingxingZhang/dep_parse/experiments/we_select_ft_pos_chinese/ft/model_0.001.std.ft0.lpos.dp.t7'
  local mst_post = MSTPostProcessor()
  mst_post:load(opts.modelPath)
  -- test performance on validation and test dataset
  print '==Valid Performance=='
  local vret = mst_post:validConllx(mst_post.opts.valid, opts.validout .. '.ori.dep')
  print(vret)
  print '--after post processing--'
  if opts.mstalg == 'ChuLiuEdmonds' then
    xprintln('Using ChuLiuEdmonds')
    vret = mst_post:validWithMSTPost(mst_post.opts.valid, opts.validout .. '.dep')
  elseif opts.mstalg == 'Eisner' then
    xprintln('Using Eisner')
    vret = mst_post:validWithMSTPostEisner(mst_post.opts.valid, opts.validout .. '.dep')
  else
    error(string.format('[%s] not supported!', opts.mstalg))
  end
  
  print(vret)
  print ''
  
  print '==Test Performance=='
  local tret = mst_post:validConllx(mst_post.opts.test, opts.testout .. '.ori.dep')
  print(tret)
  print '--after post processing--'
  if opts.mstalg == 'ChuLiuEdmonds' then
    xprintln('Using ChuLiuEdmonds')
    tret = mst_post:validWithMSTPost(mst_post.opts.test, opts.testout .. '.dep')
  elseif opts.mstalg == 'Eisner' then
    xprintln('Using Eisner')
    tret = mst_post:validWithMSTPostEisner(mst_post.opts.test, opts.testout .. '.dep')
  else
    error(string.format('[%s] not supported!', opts.mstalg))
  end
  print(tret)
  print ''
end

if not package.loaded['mst_postprocess'] then
  main()
end
