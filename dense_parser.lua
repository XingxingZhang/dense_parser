
require '.'
require 'shortcut'
require 'SelectNetPos'
require 'DepPosDataIter'
require 'PostDepGraph'
require 'ChuLiuEdmonds'
require 'Eisner'
require 'MLP'

local Parser = torch.class('DeNSeParser')

function Parser:showOpts()
  local tmp_vocab = self.opts.vocab
  self.opts.vocab = nil
  print(self.opts)
  self.opts.vocab = tmp_vocab
end


function Parser:load(modelPath, classifierPath)
  self.opts = torch.load( modelPath:sub(1, -3) .. 'state.t7' )
  local opts = self.opts
  -- disable loading pre-trained word embeddings
  opts.wordEmbedding = ''
  
  torch.manualSeed(opts.seed + 1)
  if opts.useGPU then
    require 'cutorch'
    require 'cunn'
    cutorch.manualSeed(opts.seed + 1)
  end
  
  self.net = SelectNetPos(opts)
  self:showOpts()
  
  xprintln('load from %s ...', modelPath)
  self.net:load(modelPath)
  xprintln('load from %s done!', modelPath)
  
  self.lbl_opts = torch.load(classifierPath:sub(1, -3) .. 'state.t7')
  self.mlp = MLP(self.lbl_opts)
  xprintln('load classifier from %s ...', modelPath)
  self.mlp:load(classifierPath)
  xprintln('load classifier from %s done!', modelPath)
  
  if self.mlp.opts.rel_vocab == nil then
    self.mlp.opts.rel_vocab = DepPosDataIter.createDepRelVocab(self.mlp.opts.inTrain)
    xprintln('load rel vocab done! You should use new version `train_lableded.lua`')
  end
end


function Parser:runChuLiuEdmonds(dsent, sent_dep, sent_graph)
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
  end
  
  return new_dsent
end


function Parser:runEisner(dsent, sent_dep, sent_graph)
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
  end
  
  return new_dsent
end


function Parser:parseConllx(inputFile, outputFile, postAlg)
  local dataIter = DepPosDataIter.createBatch(self.opts.vocab, inputFile, self.opts.batchSize, self.opts.maxTrainLen)
  local totalCnt = 0
  local totalLoss = 0
  local cnt = 0
  
  local fout = io.open(outputFile, 'w')
  local y_tmp = torch.LongTensor(self.opts.maxTrainLen, self.opts.batchSize)
  local cls_in_dim = 4 * self.opts.nhid + 2 * self.opts.npin + 2 * self.opts.nin
  local cls_in = torch.CudaTensor(self.opts.maxTrainLen * self.opts.batchSize, cls_in_dim)
  local dep_iter = DepPosDataIter.conllx_iter(inputFile)
  
  for x, x_mask, x_pos, y in dataIter do
    local loss, y_preds = self.net:validBatch(x, x_mask, x_pos, y)
    local x_emb = self.net.mod_map.forward_lookup:forward(x)
    local x_pos_emb = self.net.mod_map.forward_pos_lookup:forward(x_pos)
    local fwd_bak_hs = self.net.all_fwd_bak_hs
    
    totalLoss = totalLoss + loss * x:size(2)
    local y_mask = x_mask[{ {2, -1}, {} }]
    
    local y_p = y_tmp:resize(y:size(1), y:size(2))
    -- WARNING: y_preds start from 2!
    for t = 2, x:size(1) do
      local _, mi = y_preds[t]:max(2)
      if self.opts.useGPU then mi = mi:double() end
      y_p[{ t-1, {} }] = mi
    end
    
    -- get labeled output (bs, seqlen, dim)
    cls_in:resize(x:size(2), x:size(1)-1, cls_in_dim):zero()
    
    -- collects sentence dependents
    -- and graph answers
    local new_dsents = {}
    for i = 1, y_mask:size(2) do
      local slen = y_mask[{ {}, i }]:sum()
      if slen > 0 then
        local dsent = dep_iter()
        local sent_dep = {}
        local sent_graph = {}
        for j = 1, slen do
          sent_dep[#sent_dep + 1] = y_p[{ j, i }] - 1
          local tmp = y_preds[j+1][{ i, {1, slen + 1} }]:double()
          sent_graph[j] = tmp
        end
        
        -- run post-processing algorithm
        assert(#sent_dep == #dsent)
        assert(#sent_graph == #dsent)
        local new_dsent
        if postAlg == 'ChuLiuEdmonds' then
          new_dsent = self:runChuLiuEdmonds(dsent, sent_dep, sent_graph)
        elseif postAlg == 'Eisner' then
          new_dsent = self:runEisner(dsent, sent_dep, sent_graph)
        else
          error('only support ChuLiuEdmonds and Eisner')
        end
        
        -- prepare labeled input
        for j, ditem in ipairs(new_dsent) do
          local parent_id = ditem.p2 + 1
          local start = 1
          cls_in[{ i, j, {start, 2 * self.opts.nhid + start - 1} }] = fwd_bak_hs[{ i, j+1, {} }]
          start = start + 2 * self.opts.nhid
          cls_in[{ i, j, {start, 2 * self.opts.nhid + start - 1} }] = fwd_bak_hs[{ i, parent_id, {} }]
          start = start + 2 * self.opts.nhid
          cls_in[{ i, j, {start, self.opts.nin * 2 + start - 1} }] = 
            torch.cat({x_emb[{ j+1, i, {} }], x_emb[{ parent_id, i, {} }]}, 1)
          start = start + 2 * self.opts.nin
          cls_in[{ i, j, {start, self.opts.npin * 2 + start - 1} }] = 
            torch.cat({x_pos_emb[{ j+1, i, {} }], x_pos_emb[{ parent_id, i, {} }]}, 1)
        end
        
        new_dsents[#new_dsents + 1] = new_dsent
      end
    end
    
    -- run labeld classifier
    local labels_ = self.mlp:predictLabelBatch( 
      cls_in:view( cls_in:size(1) * cls_in:size(2), cls_in:size(3) )
    )
    local labels = labels_:view( cls_in:size(1), cls_in:size(2) )
    -- output everything!
    for i, dsent in ipairs(new_dsents) do
      for j, ditem in ipairs(dsent) do
        -- 1	Influential	_	JJ	JJ	_	2	amod	_	_
        local lbl = self.mlp.opts.rel_vocab.idx2rel[ labels[{ i, j }] ]
        fout:write( string.format('%d\t%s\t_\t_\t%s\t_\t%d\t%s\t_\t_\n', ditem.p1, ditem.wd, ditem.pos, ditem.p2, lbl) )
      end
      fout:write('\n')
    end
    
    totalCnt = totalCnt + y_mask:sum()
    cnt = cnt + 1
    if cnt % 5 == 0 then
      collectgarbage()
      xprintln('cnt = %d * %d = %d', cnt, self.opts.batchSize, cnt * self.opts.batchSize)
    end
  end
  fout:close()
end


local function getOpts()
  local cmd = torch.CmdLine()
  cmd:option('--modelPath', '/disk/scratch/s1270921/dep_parse/experiments/pre-trained-models/german/model_0.001.tune.t7', 'model path')
  cmd:option('--classifierPath', '/disk/scratch/s1270921/dep_parse/experiments/pre-trained-models/german/lbl_classifier.t7', 'label classifer path')
  
  cmd:option('--input', '/disk/scratch/s1270921/dep_parse/data_conll/german/german_gold_test.conll', 'input conllx file')
  cmd:option('--output', 'output.txt', 'output conllx file')
  cmd:option('--gold', '', 'gold standard file (optional). Empty means no evaluation')
  cmd:option('--mstalg', 'ChuLiuEdmonds', 'MST algorithm: ChuLiuEdmonds or Eisner')
  
  return cmd:parse(arg)
end


local function main()
  local opts = getOpts()
  local dense = DeNSeParser()
  dense:load(opts.modelPath, opts.classifierPath)
  dense:parseConllx(opts.input, opts.output, opts.mstalg)
  
  if opts.gold ~= '' or opts.gold == nil then
    print '\n\n*** Stanford ***'
    local conllx_eval = require 'conllx_eval'
    local LAS, UAS, noPunctLAS, noPunctUAS = conllx_eval.eval(opts.output, opts.gold)
    
    print '\n\n*** CoNLL-X 2006 ***'
    conllx_eval = require 'conllx2006_eval'
    LAS, UAS, noPunctLAS, noPunctUAS = conllx_eval.eval(opts.output, opts.gold)
  end
end


if not package.loaded['dense_parser'] then
  main()
else
  print '[dense_parser] loaded as package!'
end

